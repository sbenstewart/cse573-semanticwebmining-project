#!/usr/bin/env python3
"""End-to-end KG build orchestrator.

Usage::

    # Default: funding pass on known-good docs only (fast sanity check)
    python scripts/run_kg_build.py --pass funding --filter known-funding

    # Funding pass, first 15 Google News docs only
    python scripts/run_kg_build.py --pass funding --filter news --limit 15

    # Funding pass, full corpus
    python scripts/run_kg_build.py --pass funding

    # Dry run: extract but don't write to Neo4j
    python scripts/run_kg_build.py --pass funding --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import load_docs
from src.kg.extractor import FundingExtractor, ProductTechExtractor
from src.kg.ingester import FundingIngester, ProductTechIngester
from src.kg.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_kg_build")

# Substrings that reliably identify known-good funding test docs in
# the sample corpus. Used by --filter known-funding.
KNOWN_FUNDING_MARKERS = ("Replit grabs", "Oska Health")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the TrendScout KG")
    p.add_argument("--pass", dest="pass_name", choices=["funding", "products"], default="funding",
                   help="Which extraction pass to run (more passes in later steps)")
    p.add_argument("--filter", choices=["known-funding", "news", "all"], default="all",
                   help="Which subset of the corpus to process")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N docs (after filtering)")
    p.add_argument("--dry-run", action="store_true",
                   help="Extract and print, but don't write to Neo4j")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model to use")
    return p.parse_args()


def select_docs(docs: list[dict], filter_name: str, limit: int | None) -> list[dict]:
    if filter_name == "known-funding":
        out = [d for d in docs
               if any(m in (d.get("title") or "") for m in KNOWN_FUNDING_MARKERS)]
    elif filter_name == "news":
        out = [d for d in docs if "News" in (d.get("publisher") or "")]
    else:
        out = list(docs)
    if limit is not None:
        out = out[:limit]
    return out


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 55)
    print(f"  TrendScout AI 2.0 - KG Build  [pass: {args.pass_name}]")
    print("=" * 55)

    docs = load_docs()
    if not docs:
        print("  No documents found. Run preprocessing first.")
        sys.exit(1)
    print(f"  Loaded {len(docs)} documents from corpus")

    selected = select_docs(docs, args.filter, args.limit)
    print(f"  Filter '{args.filter}' matched {len(selected)} docs"
          + (f" (limited to {args.limit})" if args.limit else ""))
    if not selected:
        print("  No documents to process. Exiting.")
        return

    if args.pass_name == "products":
        extractor = ProductTechExtractor(model=args.model)
    else:
        extractor = FundingExtractor(model=args.model)

    if args.dry_run:
        print(f"\n  [DRY RUN] Extracting only, not writing to Neo4j\n")
        total = 0
        for i, doc in enumerate(selected, 1):
            print(f"  [{i}/{len(selected)}] {(doc.get('title') or '')[:70]}")
            results = extractor.extract(doc)
            for r in results:
                if args.pass_name == "products":
                    print(f"      → {r.company}  ::  {r.product}  techs={r.technologies}")
                else:
                    print(f"      → {r.company}  {r.amount_raw}  "
                          f"({r.round_type})  investors={r.investors}")
            total += len(results)
        print(f"\n  Total items extracted: {total}")
        return

    with Neo4jClient() as client:
        if args.pass_name == "products":
            ingester = ProductTechIngester(client)
        else:
            ingester = FundingIngester(client)
        total = 0
        for i, doc in enumerate(selected, 1):
            title_snippet = (doc.get("title") or "")[:70]
            print(f"  [{i}/{len(selected)}] {title_snippet}")

            if args.pass_name == "funding":
                ingester.upsert_document(doc)

            results = extractor.extract(doc)
            if not results:
                print(f"      (no items found)")
                continue

            if args.pass_name == "products":
                n_written = ingester.ingest_mentions(results, doc)
                for r in results:
                    print(f"      → {r.company}  ::  {r.product}  techs={r.technologies}")
            else:
                n_written = ingester.ingest_rounds(results, doc)
                for r in results:
                    print(f"      → {r.company}  {r.amount_raw}  "
                          f"({r.round_type})  investors={r.investors}")
            total += n_written

        print(f"\n  Total items ingested: {total}")
        print(f"  Graph now contains:")
        print(f"    Nodes        : {client.node_count()}")
        print(f"    Relationships: {client.relationship_count()}")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
