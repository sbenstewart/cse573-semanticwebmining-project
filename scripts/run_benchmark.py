#!/usr/bin/env python3
"""Phase 4 benchmark runner.

Loads the eval question set, runs both Approach A (text-to-Cypher) and
Approach B (GraphRAG) over every question, scores the answers, and saves
the full results to a JSON file for analysis.

Usage:
    # Run full benchmark (default — all 25 questions × both approaches)
    python scripts/run_benchmark.py

    # Single approach
    python scripts/run_benchmark.py --approach A
    python scripts/run_benchmark.py --approach B

    # Subset for fast iteration (first 5 questions)
    python scripts/run_benchmark.py --limit 5

    # Custom output path
    python scripts/run_benchmark.py --output evaluation/results_v1.json

Estimated runtime: ~15-25 minutes for the full 50-call benchmark
on llama3.1:8b via local Ollama. Each call is 5-20 seconds depending
on model warmth and question complexity.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.scorer import (
    QuestionScore,
    aggregate_scores,
    score_answer,
)
from src.kg.neo4j_client import Neo4jClient
from src.rag.common import Answer

logging.basicConfig(
    level=logging.WARNING,  # quiet — we print our own progress
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict]:
    """Load and validate the YAML question set."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in {path}")

    # Validate each question has the required fields
    for q in questions:
        for field in ("id", "category", "question"):
            if field not in q:
                raise ValueError(f"Question missing '{field}': {q}")
        q.setdefault("expected_keywords", [])
        q.setdefault("expected_doc_ids", None)
        q.setdefault("notes", "")
    return questions


def run_one_approach(
    system,
    questions: list[dict],
    approach_label: str,
) -> tuple[list[Answer], list[QuestionScore]]:
    """Run one QA system over all questions, scoring as we go."""
    answers: list[Answer] = []
    scores: list[QuestionScore] = []

    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {approach_label} ({len(questions)} questions)")
    print(f"{'=' * 60}")

    for i, q in enumerate(questions, 1):
        print(
            f"\n[{i:2d}/{len(questions)}] [{q['category']:14s}] "
            f"{q['id']}: {q['question'][:60]}",
            flush=True,
        )

        t_start = time.perf_counter()
        try:
            answer = system.answer(q["question"])
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            answer = Answer(
                question=q["question"], text="",
                approach=approach_label, latency_ms=elapsed_ms,
                error=f"Pipeline raised {type(e).__name__}: {e}",
            )
        elapsed_s = (time.perf_counter() - t_start)

        answers.append(answer)
        score = score_answer(q, answer)
        scores.append(score)

        # Print one-line summary
        if score.answered:
            cov_str = f"keywords={score.keyword_coverage:.0%}"
            preview = (answer.text or "")[:80].replace("\n", " ")
            print(f"          ✓ {cov_str} ({elapsed_s:.1f}s) {preview}...")
        else:
            err_short = (answer.error or "")[:80].replace("\n", " ")
            print(f"          ✗ ERROR ({elapsed_s:.1f}s) {err_short}")

    return answers, scores


def save_results(
    output_path: Path,
    questions: list[dict],
    all_answers: dict[str, list[Answer]],
    all_scores: list[QuestionScore],
    aggregates: dict[str, dict],
    metadata: dict,
) -> None:
    """Serialize the full benchmark output as JSON."""
    payload = {
        "metadata": metadata,
        "questions": questions,
        "answers": {
            approach: [a.to_dict() for a in answers]
            for approach, answers in all_answers.items()
        },
        "scores": [s.to_dict() for s in all_scores],
        "aggregates": aggregates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Saved results to {output_path}")


def print_summary(aggregates: dict[str, dict]) -> None:
    """Print a compact summary table to the terminal."""
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'=' * 60}\n")

    if not aggregates:
        print("  No results.")
        return

    approaches = list(aggregates.keys())

    # Top-level metrics
    print(f"  {'Metric':<28s} " + " ".join(f"{a:>14s}" for a in approaches))
    print(f"  {'-' * 28} " + " ".join("-" * 14 for _ in approaches))

    metrics = [
        ("N questions", "n_questions", lambda v: f"{v}"),
        ("N answered", "n_answered", lambda v: f"{v}"),
        ("Answered rate", "answered_rate", lambda v: f"{v:.0%}"),
        ("Mean keyword coverage", "mean_keyword_coverage", lambda v: f"{v:.0%}"),
        ("Mean latency (s)", "mean_latency_ms", lambda v: f"{v / 1000:.1f}"),
        ("Median latency (s)", "median_latency_ms", lambda v: f"{v / 1000:.1f}"),
        ("Mean answer length", "mean_answer_length", lambda v: f"{v:.0f}"),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:<28s} " + " ".join(
            f"{fmt(aggregates[a][key]):>14s}" for a in approaches
        )
        print(row)

    # Per-category coverage
    print(f"\n  Per-category keyword coverage:")
    print(f"  {'Category':<28s} " + " ".join(f"{a:>14s}" for a in approaches))
    print(f"  {'-' * 28} " + " ".join("-" * 14 for _ in approaches))

    # Get sorted union of categories
    all_categories = set()
    for a in approaches:
        all_categories.update(aggregates[a]["per_category"].keys())

    for cat in sorted(all_categories):
        row_cells = []
        for a in approaches:
            pc = aggregates[a]["per_category"].get(cat, {})
            cov = pc.get("mean_keyword_coverage", 0.0)
            n_ans = pc.get("n_answered", 0)
            n_tot = pc.get("n", 0)
            row_cells.append(f"{cov:.0%} ({n_ans}/{n_tot})")
        row = f"  {cat:<28s} " + " ".join(f"{c:>14s}" for c in row_cells)
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 benchmark — A vs B comparison",
    )
    parser.add_argument(
        "--questions", type=str,
        default="evaluation/eval_questions.yaml",
        help="Path to eval question YAML",
    )
    parser.add_argument(
        "--approach", choices=["A", "B", "both"], default="both",
        help="Which approach(es) to benchmark",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N questions (for quick iteration)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Run only questions of this category",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: evaluation/results_<timestamp>.json)",
    )
    args = parser.parse_args()

    # Load questions
    questions = load_questions(Path(args.questions))
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    print(f"\n  Loaded {len(questions)} questions from {args.questions}")
    if args.category:
        print(f"  Filtered to category: {args.category}")
    if args.limit:
        print(f"  Limited to first: {args.limit}")

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"evaluation/results_{ts}.json")

    # Build the systems we need
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(questions),
        "approach": args.approach,
        "questions_path": str(args.questions),
    }

    all_answers: dict[str, list[Answer]] = {}
    all_scores: list[QuestionScore] = []

    benchmark_start = time.perf_counter()

    with Neo4jClient() as client:
        if args.approach in ("A", "both"):
            from src.rag.text_to_cypher import TextToCypherQA
            system_a = TextToCypherQA(neo4j_client=client)
            answers_a, scores_a = run_one_approach(
                system_a, questions, "text_to_cypher"
            )
            all_answers["text_to_cypher"] = answers_a
            all_scores.extend(scores_a)

        if args.approach in ("B", "both"):
            from src.rag.graph_rag import GraphRAG
            system_b = GraphRAG(neo4j_client=client)
            answers_b, scores_b = run_one_approach(
                system_b, questions, "graph_rag"
            )
            all_answers["graph_rag"] = answers_b
            all_scores.extend(scores_b)

    total_elapsed = time.perf_counter() - benchmark_start
    metadata["total_runtime_seconds"] = total_elapsed

    # Compute aggregates
    aggregates = {}
    for approach in all_answers:
        agg = aggregate_scores(all_scores, approach)
        aggregates[approach] = agg.to_dict()

    # Save and report
    save_results(
        output_path, questions, all_answers, all_scores, aggregates, metadata
    )
    print_summary(aggregates)
    print(f"\n  Total benchmark runtime: {total_elapsed / 60:.1f} minutes")
    print(f"  Full results: {output_path}\n")


if __name__ == "__main__":
    main()
