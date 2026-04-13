#!/usr/bin/env python3
"""Initialize the Neo4j knowledge graph schema.

Idempotent: safe to run repeatedly. On each run it applies all uniqueness
constraints and indexes, then prints a summary of what's currently in the
database.

Usage::

    python scripts/run_kg_init.py           # apply schema, print summary
    python scripts/run_kg_init.py --wipe    # DELETE ALL DATA, then apply schema
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.neo4j_client import Neo4jClient
from src.kg.schema import apply_schema, describe_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_kg_init")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize the TrendScout KG schema")
    p.add_argument(
        "--wipe",
        action="store_true",
        help="Delete ALL nodes and relationships before applying the schema. "
             "Use this to start fresh.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 55)
    print("  TrendScout AI 2.0 - KG Schema Initialization")
    print("=" * 55)

    with Neo4jClient() as client:
        if args.wipe:
            print("\n  [!] --wipe requested: deleting all existing data...")
            client.wipe()

        print("\n  Applying schema...")
        summary = apply_schema(client)
        print(f"  ✅ {summary['constraints']} constraints applied")
        print(f"  ✅ {summary['indexes']} indexes applied")

        schema_info = describe_schema(client)
        print(f"\n  Current schema:")
        print(f"    Constraints in database: {len(schema_info['constraints'])}")
        print(f"    Indexes in database    : {len(schema_info['indexes'])}")

        n_nodes = client.node_count()
        n_rels = client.relationship_count()
        print(f"\n  Current data:")
        print(f"    Nodes        : {n_nodes}")
        print(f"    Relationships: {n_rels}")

    print("\n  Done. Open http://localhost:7474 to inspect.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
