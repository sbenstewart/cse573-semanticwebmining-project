#!/usr/bin/env python3
"""Surgical cleanup of obvious extraction noise in the KG.

After Pass 2 the graph contains some garbage Product nodes that the LLM
extracted from descriptive boilerplate ("Scale AI's product (no specific
name mentioned)", long descriptive sentences, etc.). This script runs a
small set of deterministic Cypher DELETEs to drop the obvious junk
without re-running extraction.

The cleanup is conservative — it only drops nodes matching unambiguous
junk patterns. Ambiguous cases are kept (we'd rather leave noise than
delete a real product).

Usage::

    python scripts/clean_kg.py --dry-run    # show what would be deleted
    python scripts/clean_kg.py              # actually delete
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.WARNING)


# Each rule has a name, a SELECT (read-only preview) query, and a DELETE query.
# The SELECT shows what would be removed in --dry-run mode; the DELETE actually
# removes it. They're paired so the preview and the action stay in sync.
CLEANUP_RULES: list[dict] = [
    {
        "name": "Parenthetical disclaimers",
        "description": "Products with names like '(not specified)' or '(no specific name mentioned)'",
        "select": """
            MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS "not specified"
               OR toLower(p.name) CONTAINS "no specific name"
               OR toLower(p.name) CONTAINS "no specific product"
               OR toLower(p.name) CONTAINS "unspecified"
            RETURN p.name AS name
            ORDER BY p.name
        """,
        "delete": """
            MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS "not specified"
               OR toLower(p.name) CONTAINS "no specific name"
               OR toLower(p.name) CONTAINS "no specific product"
               OR toLower(p.name) CONTAINS "unspecified"
            DETACH DELETE p
        """,
    },
    {
        "name": "Long descriptive sentences",
        "description": "Product names longer than 8 words (almost always descriptions, not names)",
        "select": """
            MATCH (p:Product)
            WHERE size(split(trim(p.name), " ")) > 8
            RETURN p.name AS name
            ORDER BY size(split(trim(p.name), " ")) DESC
        """,
        "delete": """
            MATCH (p:Product)
            WHERE size(split(trim(p.name), " ")) > 8
            DETACH DELETE p
        """,
    },
    {
        "name": "Product name equals a Startup name",
        "description": "Products that are actually company names (e.g. 'Scale AI' as a Product)",
        "select": """
            MATCH (p:Product), (s:Startup)
            WHERE p.normalized_name = s.normalized_name
            RETURN p.name AS name
            ORDER BY p.name
        """,
        "delete": """
            MATCH (p:Product), (s:Startup)
            WHERE p.normalized_name = s.normalized_name
            DETACH DELETE p
        """,
    },
    {
        "name": "Possessive 'X's product/platform' patterns",
        "description": "Names like 'Scale AI's platform', 'Replit's product'",
        "select": """
            MATCH (p:Product)
            WHERE toLower(p.name) =~ ".*'s (product|platform|tool|service|offering|solution).*"
            RETURN p.name AS name
            ORDER BY p.name
        """,
        "delete": """
            MATCH (p:Product)
            WHERE toLower(p.name) =~ ".*'s (product|platform|tool|service|offering|solution).*"
            DETACH DELETE p
        """,
    },
    {
        "name": "Bare generic product names",
        "description": "Products like 'AI agents', 'agentic AI products', 'GenAI solutions'",
        "select": """
            MATCH (p:Product)
            WHERE toLower(p.name) IN [
                "ai agents", "agentic ai products", "agentic ai platforms",
                "genai solutions", "scale products", "ai platform",
                "the platform", "the product", "frontier ai",
                "frontier agentic data products", "llms", "llms (large language models)",
                "llm gateway service"
            ]
            RETURN p.name AS name
            ORDER BY p.name
        """,
        "delete": """
            MATCH (p:Product)
            WHERE toLower(p.name) IN [
                "ai agents", "agentic ai products", "agentic ai platforms",
                "genai solutions", "scale products", "ai platform",
                "the platform", "the product", "frontier ai",
                "frontier agentic data products", "llms", "llms (large language models)",
                "llm gateway service"
            ]
            DETACH DELETE p
        """,
    },
    {
        "name": "Orphan Technology nodes",
        "description": "Technology nodes left with no Product after Product cleanup",
        "select": """
            MATCH (t:Technology)
            WHERE NOT (t)<-[:USES_TECH]-(:Product)
            RETURN t.name AS name
            ORDER BY t.name
        """,
        "delete": """
            MATCH (t:Technology)
            WHERE NOT (t)<-[:USES_TECH]-(:Product)
            DETACH DELETE t
        """,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean obvious extraction noise from the KG")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be deleted, but don't delete")
    return p.parse_args()


def print_rule_preview(client: Neo4jClient, rule: dict) -> int:
    rows = client.run_read(rule["select"])
    print(f"\n  ▸ {rule['name']}")
    print(f"    {rule['description']}")
    print(f"    Matched: {len(rows)} node(s)")
    if rows:
        for r in rows[:15]:
            name = r.get("name", "")
            print(f"      - {name[:90]}")
        if len(rows) > 15:
            print(f"      ... and {len(rows) - 15} more")
    return len(rows)


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print(f"  TrendScout AI 2.0 - KG Cleanup  "
          f"{'[DRY RUN]' if args.dry_run else '[LIVE]'}")
    print("=" * 60)

    with Neo4jClient() as client:
        before_nodes = client.node_count()
        before_rels = client.relationship_count()
        print(f"\n  Before:  {before_nodes} nodes, {before_rels} relationships")

        total_matched = 0
        for rule in CLEANUP_RULES:
            n = print_rule_preview(client, rule)
            total_matched += n
            if not args.dry_run and n > 0:
                client.run_write(rule["delete"])

        if args.dry_run:
            print(f"\n  Would delete: {total_matched} node(s) total")
            print("  Re-run without --dry-run to actually delete.")
        else:
            after_nodes = client.node_count()
            after_rels = client.relationship_count()
            print(f"\n  After:   {after_nodes} nodes, {after_rels} relationships")
            print(f"  Removed: {before_nodes - after_nodes} nodes, "
                  f"{before_rels - after_rels} relationships")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
