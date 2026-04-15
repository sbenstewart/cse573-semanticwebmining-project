#!/usr/bin/env python3
"""Interactive KG-RAG chat for TrendScout AI 2.0.

Supports both Approach A (text-to-Cypher) and Approach B (GraphRAG).
Run interactively or in one-shot mode.

Usage:
    # Interactive REPL (default: both approaches)
    python scripts/run_qa_chat.py

    # Single approach
    python scripts/run_qa_chat.py --approach A
    python scripts/run_qa_chat.py --approach B

    # One-shot mode
    python scripts/run_qa_chat.py --query "Who invested in Replit?"

    # One-shot with specific approach
    python scripts/run_qa_chat.py --approach A --query "Who invested in Replit?"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.neo4j_client import Neo4jClient
from src.rag.common import Answer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)


def _print_answer(ans: Answer) -> None:
    """Pretty-print an Answer object."""
    label = ans.approach.upper().replace("_", " ")
    print(f"\n  [{label}]")
    if ans.is_error():
        print(f"  ERROR: {ans.error}")
    else:
        # Word-wrap the answer at ~78 chars
        words = ans.text.split()
        lines, current = [], ""
        for w in words:
            if len(current) + len(w) + 1 > 78:
                lines.append(current)
                current = w
            else:
                current = f"{current} {w}" if current else w
        if current:
            lines.append(current)
        for line in lines:
            print(f"  {line}")

    print(f"  ({ans.latency_ms:.0f}ms)")

    if ans.cited_doc_ids:
        print(f"  Sources: {len(ans.cited_doc_ids)} document(s)")

    # Show generated Cypher for Approach A
    if ans.approach == "text_to_cypher" and ans.trace.get("attempts"):
        last = ans.trace["attempts"][-1]
        if last.get("cypher"):
            cypher_preview = last["cypher"][:120]
            print(f"  Cypher: {cypher_preview}...")

    # Show retrieval info for Approach B
    if ans.approach == "graph_rag" and ans.trace.get("retrieved_docs"):
        docs = ans.trace["retrieved_docs"]
        print(f"  Retrieved {len(docs)} documents:")
        for d in docs[:3]:
            score = d.get("score", 0)
            title = (d.get("title") or "?")[:70]
            print(f"    [{score:.3f}] {title}")


def run_one_shot(question: str, approach: str, client: Neo4jClient) -> None:
    """Answer a single question and exit."""
    systems = _build_systems(approach, client)
    for system in systems:
        ans = system.answer(question)
        _print_answer(ans)


def run_repl(approach: str, client: Neo4jClient) -> None:
    """Run the interactive REPL."""
    print("\n" + "=" * 60)
    print("  TrendScout AI 2.0 — KG-RAG Chat (Phase 3)")
    print("=" * 60)
    mode = {
        "A": "Approach A (text-to-Cypher)",
        "B": "Approach B (GraphRAG)",
        "both": "Both approaches (side-by-side)",
    }[approach]
    print(f"  Mode: {mode}")
    print(f"  Type 'quit' or 'exit' to leave.")
    print("=" * 60)

    systems = _build_systems(approach, client)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        for system in systems:
            ans = system.answer(question)
            _print_answer(ans)


def _build_systems(approach: str, client: Neo4jClient) -> list:
    """Instantiate the requested QA system(s)."""
    systems = []

    if approach in ("A", "both"):
        from src.rag.text_to_cypher import TextToCypherQA
        systems.append(TextToCypherQA(neo4j_client=client))

    if approach in ("B", "both"):
        from src.rag.graph_rag import GraphRAG
        systems.append(GraphRAG(neo4j_client=client))

    return systems


def main():
    parser = argparse.ArgumentParser(
        description="TrendScout AI 2.0 — KG-RAG Chat"
    )
    parser.add_argument(
        "--approach", choices=["A", "B", "both"], default="both",
        help="Which QA approach to use (default: both)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="One-shot query (skip REPL)",
    )
    args = parser.parse_args()

    with Neo4jClient() as client:
        if args.query:
            run_one_shot(args.query, args.approach, client)
        else:
            run_repl(args.approach, client)


if __name__ == "__main__":
    main()
