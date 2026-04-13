#!/usr/bin/env python3
"""Demo Cypher queries for the TrendScout KG (Phase 2).

Runs a canned set of questions against Neo4j and pretty-prints the answers.
Use this as a sanity check after ingestion, and as a reference for the kinds
of questions the KG can answer. Invoke without args to run every query, or
pass --query <name> to run one.

Usage::

    python scripts/run_kg_queries.py              # run all
    python scripts/run_kg_queries.py --list       # list available queries
    python scripts/run_kg_queries.py --query top-rounds
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.WARNING)


# --- query catalog --------------------------------------------------------

QUERIES: dict[str, dict] = {
    "summary": {
        "title": "Graph summary — nodes and relationships by type",
        "cypher": """
            MATCH (n)
            WITH labels(n)[0] AS label, count(*) AS n
            RETURN label, n ORDER BY n DESC
        """,
        "headers": ["Label", "Count"],
        "format": lambda r: [r["label"], r["n"]],
    },
    "rel-summary": {
        "title": "Relationships by type",
        "cypher": """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(*) AS n ORDER BY n DESC
        """,
        "headers": ["Relationship", "Count"],
        "format": lambda r: [r["rel_type"], r["n"]],
    },
    "top-rounds": {
        "title": "Top 10 largest funding rounds",
        "cypher": """
            MATCH (s:Startup)-[:HAS_FUNDING_ROUND]->(r:FundingRound)
            RETURN s.name AS startup,
                   r.amount_raw AS amount,
                   r.amount_usd AS amount_usd,
                   r.round_type AS round_type,
                   r.announced_date AS date
            ORDER BY r.amount_usd DESC
            LIMIT 10
        """,
        "headers": ["Startup", "Amount", "USD", "Round", "Date"],
        "format": lambda r: [
            r["startup"], r["amount"] or "—",
            f"${r['amount_usd']:,}" if r["amount_usd"] else "—",
            r["round_type"] or "—", r["date"] or "—",
        ],
    },
    "rounds-by-type": {
        "title": "Funding rounds grouped by round type",
        "cypher": """
            MATCH (r:FundingRound)
            RETURN r.round_type AS round_type,
                   count(*) AS n,
                   sum(r.amount_usd) AS total_usd
            ORDER BY n DESC
        """,
        "headers": ["Round Type", "Count", "Total Raised"],
        "format": lambda r: [
            r["round_type"] or "Unknown",
            r["n"],
            f"${r['total_usd']:,}" if r["total_usd"] else "—",
        ],
    },
    "top-investors": {
        "title": "Most active investors (by rounds participated in)",
        "cypher": """
            MATCH (i:Investor)<-[:INVESTED_BY]-(r:FundingRound)
            RETURN i.name AS investor, count(r) AS rounds
            ORDER BY rounds DESC
            LIMIT 10
        """,
        "headers": ["Investor", "# Rounds"],
        "format": lambda r: [r["investor"], r["rounds"]],
    },
    "investor-portfolio": {
        "title": "For each investor: which startups they funded",
        "cypher": """
            MATCH (i:Investor)<-[:INVESTED_BY]-(r:FundingRound)<-[:HAS_FUNDING_ROUND]-(s:Startup)
            RETURN i.name AS investor, collect(DISTINCT s.name) AS portfolio
            ORDER BY size(collect(DISTINCT s.name)) DESC
            LIMIT 10
        """,
        "headers": ["Investor", "Portfolio"],
        "format": lambda r: [r["investor"], ", ".join(r["portfolio"])],
    },
    "rounds-with-provenance": {
        "title": "Funding rounds with source document titles",
        "cypher": """
            MATCH (s:Startup)-[:HAS_FUNDING_ROUND]->(r:FundingRound)-[:SOURCED_FROM]->(d:Document)
            RETURN s.name AS startup,
                   r.amount_raw AS amount,
                   d.title AS source_title,
                   d.publisher AS publisher
            ORDER BY r.amount_usd DESC
            LIMIT 10
        """,
        "headers": ["Startup", "Amount", "Source Title", "Publisher"],
        "format": lambda r: [
            r["startup"], r["amount"] or "—",
            (r["source_title"] or "")[:60], r["publisher"] or "—",
        ],
    },
    "orphan-rounds": {
        "title": "Rounds with no investors (data-quality check)",
        "cypher": """
            MATCH (s:Startup)-[:HAS_FUNDING_ROUND]->(r:FundingRound)
            WHERE NOT (r)-[:INVESTED_BY]->(:Investor)
            RETURN s.name AS startup,
                   r.amount_raw AS amount,
                   r.round_type AS round_type
            ORDER BY r.amount_usd DESC
        """,
        "headers": ["Startup", "Amount", "Round"],
        "format": lambda r: [r["startup"], r["amount"] or "—", r["round_type"] or "—"],
    },
    "products": {
        "title": "Named products extracted (Pass 2)",
        "cypher": """
            MATCH (s:Startup)-[:ANNOUNCED]->(p:Product)
            OPTIONAL MATCH (p)-[:USES_TECH]->(t:Technology)
            RETURN s.name AS startup,
                   p.name AS product,
                   p.description AS description,
                   collect(DISTINCT t.name) AS techs
            ORDER BY startup, product
        """,
        "headers": ["Startup", "Product", "Tech Stack", "Description"],
        "format": lambda r: [
            r["startup"], r["product"],
            ", ".join(r["techs"]) if r["techs"] else "—",
            (r["description"] or "")[:50],
        ],
    },
    "top-tech": {
        "title": "Most frequently mentioned technologies",
        "cypher": """
            MATCH (t:Technology)<-[:USES_TECH]-(p:Product)
            RETURN t.name AS tech, count(DISTINCT p) AS products
            ORDER BY products DESC
            LIMIT 15
        """,
        "headers": ["Technology", "# Products"],
        "format": lambda r: [r["tech"], r["products"]],
    },
    "funded-and-shipping": {
        "title": "Startups that BOTH raised funding AND announced products",
        "cypher": """
            MATCH (s:Startup)-[:HAS_FUNDING_ROUND]->(r:FundingRound)
            MATCH (s)-[:ANNOUNCED]->(p:Product)
            RETURN s.name AS startup,
                   collect(DISTINCT p.name) AS products,
                   sum(r.amount_usd) AS total_raised
            ORDER BY total_raised DESC
        """,
        "headers": ["Startup", "Products", "Total Raised"],
        "format": lambda r: [
            r["startup"],
            ", ".join(r["products"]),
            f"${r['total_raised']:,}" if r["total_raised"] else "—",
        ],
    },
}


# --- pretty printing ------------------------------------------------------

def print_table(headers: list[str], rows: list[list]) -> None:
    if not rows:
        print("    (no results)")
        return
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(h), max((len(row[i]) for row in str_rows), default=0))
        for i, h in enumerate(headers)
    ]
    line = "  | " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "  |" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(line)
    print(sep)
    for row in str_rows:
        print("  | " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")


def run_query(client: Neo4jClient, name: str, spec: dict) -> None:
    print(f"\n  {spec['title']}")
    print("  " + "─" * (len(spec["title"]) + 2))
    rows = client.run_read(spec["cypher"])
    formatter: Callable = spec["format"]
    table_rows = [formatter(r) for r in rows]
    print_table(spec["headers"], table_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run demo KG queries")
    p.add_argument("--query", help="Run a single named query (see --list)")
    p.add_argument("--list", action="store_true", help="List available queries")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("\nAvailable queries:")
        for name, spec in QUERIES.items():
            print(f"  {name:<24}  {spec['title']}")
        print()
        return

    print("\n" + "=" * 60)
    print("  TrendScout AI 2.0 - KG Demo Queries")
    print("=" * 60)

    with Neo4jClient() as client:
        if args.query:
            if args.query not in QUERIES:
                print(f"  Unknown query '{args.query}'. Use --list to see options.")
                sys.exit(1)
            run_query(client, args.query, QUERIES[args.query])
        else:
            for name, spec in QUERIES.items():
                run_query(client, name, spec)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
