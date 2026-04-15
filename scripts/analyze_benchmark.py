#!/usr/bin/env python3
"""Pretty-print Phase 4 benchmark results as Markdown tables.

Reads the JSON output from run_benchmark.py and produces human-readable
Markdown suitable for pasting into the final report.

Usage:
    # Latest results file
    python scripts/analyze_benchmark.py

    # Specific results file
    python scripts/analyze_benchmark.py --input evaluation/results_20260415_143022.json

    # Save to file instead of stdout
    python scripts/analyze_benchmark.py --output evaluation/report.md

    # Detailed mode (per-question table with answers)
    python scripts/analyze_benchmark.py --detail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_results(eval_dir: Path = Path("evaluation")) -> Path | None:
    """Find the most recent results_*.json file."""
    candidates = sorted(eval_dir.glob("results_*.json"))
    return candidates[-1] if candidates else None


def fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def fmt_seconds(ms: float) -> str:
    return f"{ms / 1000:.1f}s"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_metadata(meta: dict) -> str:
    """Top-level run info."""
    out = ["## Benchmark Run Metadata\n"]
    out.append(f"- **Timestamp:** {meta.get('timestamp', '?')}")
    out.append(f"- **Questions:** {meta.get('n_questions', '?')}")
    out.append(f"- **Approaches benchmarked:** {meta.get('approach', '?')}")
    runtime = meta.get("total_runtime_seconds", 0)
    out.append(f"- **Total runtime:** {runtime / 60:.1f} minutes ({runtime:.0f}s)")
    return "\n".join(out) + "\n"


def section_overall_summary(aggregates: dict) -> str:
    """High-level metrics comparison."""
    out = ["## Overall Comparison\n"]
    if not aggregates:
        return out[0] + "\n*(no aggregates available)*\n"

    approaches = list(aggregates.keys())
    headers = ["Metric"] + approaches

    rows = [
        ["Questions", *[str(aggregates[a]["n_questions"]) for a in approaches]],
        ["Answered (no error)", *[str(aggregates[a]["n_answered"]) for a in approaches]],
        ["Answered rate", *[fmt_pct(aggregates[a]["answered_rate"]) for a in approaches]],
        ["Mean keyword coverage", *[fmt_pct(aggregates[a]["mean_keyword_coverage"]) for a in approaches]],
        ["Mean latency", *[fmt_seconds(aggregates[a]["mean_latency_ms"]) for a in approaches]],
        ["Median latency", *[fmt_seconds(aggregates[a]["median_latency_ms"]) for a in approaches]],
        ["Mean answer length (chars)", *[f"{aggregates[a]['mean_answer_length']:.0f}" for a in approaches]],
    ]
    out.append(md_table(headers, rows))
    return "\n".join(out) + "\n"


def section_per_category(aggregates: dict) -> str:
    """Per-category breakdown across approaches."""
    out = ["## Per-Category Performance\n"]
    approaches = list(aggregates.keys())
    if not approaches:
        return out[0] + "\n*(no aggregates)*\n"

    # Union of categories from all approaches
    all_categories = set()
    for a in approaches:
        all_categories.update(aggregates[a].get("per_category", {}).keys())
    categories = sorted(all_categories)

    # Two stacked tables: keyword coverage and answered rate
    out.append("### Keyword coverage by category\n")
    headers = ["Category"] + approaches
    rows = []
    for cat in categories:
        row = [cat]
        for a in approaches:
            pc = aggregates[a]["per_category"].get(cat, {})
            cov = pc.get("mean_keyword_coverage", 0.0)
            row.append(fmt_pct(cov))
        rows.append(row)
    out.append(md_table(headers, rows))

    out.append("\n### Answered rate by category\n")
    rows = []
    for cat in categories:
        row = [cat]
        for a in approaches:
            pc = aggregates[a]["per_category"].get(cat, {})
            rate = pc.get("answered_rate", 0.0)
            n_ans = pc.get("n_answered", 0)
            n_tot = pc.get("n", 0)
            row.append(f"{fmt_pct(rate)} ({n_ans}/{n_tot})")
        rows.append(row)
    out.append(md_table(headers, rows))

    out.append("\n### Mean latency by category\n")
    rows = []
    for cat in categories:
        row = [cat]
        for a in approaches:
            pc = aggregates[a]["per_category"].get(cat, {})
            lat = pc.get("mean_latency_ms", 0.0)
            row.append(fmt_seconds(lat))
        rows.append(row)
    out.append(md_table(headers, rows))

    return "\n".join(out) + "\n"


def section_per_question(
    questions: list[dict],
    scores: list[dict],
    answers: dict,
) -> str:
    """Per-question breakdown — all 25 questions, both approaches."""
    out = ["## Per-Question Results\n"]

    by_qid: dict[str, dict[str, dict]] = {}
    for s in scores:
        qid = s["question_id"]
        by_qid.setdefault(qid, {})[s["approach"]] = s

    headers = ["Q", "Category", "Question", "A: cov", "A: status",
               "B: cov", "B: status"]
    rows = []
    for q in questions:
        qid = q["id"]
        s_dict = by_qid.get(qid, {})

        a_score = s_dict.get("text_to_cypher", {})
        b_score = s_dict.get("graph_rag", {})

        a_cov = fmt_pct(a_score.get("keyword_coverage", 0)) if a_score else "—"
        b_cov = fmt_pct(b_score.get("keyword_coverage", 0)) if b_score else "—"

        a_status = "✓" if a_score.get("answered") else (
            "✗ err" if a_score else "—"
        )
        b_status = "✓" if b_score.get("answered") else (
            "✗ err" if b_score else "—"
        )

        question_short = q["question"][:55] + (
            "..." if len(q["question"]) > 55 else ""
        )
        rows.append([
            qid, q["category"], question_short,
            a_cov, a_status, b_cov, b_status,
        ])

    out.append(md_table(headers, rows))
    return "\n".join(out) + "\n"


def section_detailed_answers(
    questions: list[dict],
    answers: dict,
    scores: list[dict],
) -> str:
    """Full answers for every question, useful for qualitative review."""
    out = ["## Detailed Answers (per question)\n"]

    by_qid_score: dict[str, dict[str, dict]] = {}
    for s in scores:
        by_qid_score.setdefault(s["question_id"], {})[s["approach"]] = s

    answers_by_qid: dict[str, dict[str, dict]] = {}
    for approach, ans_list in answers.items():
        for q, ans in zip(questions, ans_list):
            answers_by_qid.setdefault(q["id"], {})[approach] = ans

    for q in questions:
        qid = q["id"]
        out.append(f"### {qid} [{q['category']}]")
        out.append(f"**Q:** {q['question']}")
        if q.get("expected_keywords"):
            out.append(
                f"_Expected keywords:_ {', '.join(q['expected_keywords'])}"
            )

        ans_dict = answers_by_qid.get(qid, {})
        score_dict = by_qid_score.get(qid, {})

        for approach in ("text_to_cypher", "graph_rag"):
            if approach not in ans_dict:
                continue
            ans = ans_dict[approach]
            sc = score_dict.get(approach, {})
            label = "Approach A (text-to-Cypher)" if approach == "text_to_cypher" \
                    else "Approach B (GraphRAG)"
            out.append(f"\n**{label}** "
                       f"({fmt_seconds(ans.get('latency_ms', 0))}, "
                       f"keyword coverage {fmt_pct(sc.get('keyword_coverage', 0))})")
            if ans.get("error"):
                out.append(f"> *ERROR:* `{ans['error'][:200]}`")
            else:
                # Indent answer as quote
                ans_text = (ans.get("text") or "").replace("\n", "\n> ")
                out.append(f"> {ans_text}")
        out.append("")  # blank line between questions

    return "\n".join(out) + "\n"


def section_takeaways(aggregates: dict) -> str:
    """Auto-generated key findings paragraph."""
    out = ["## Key Findings\n"]
    if "text_to_cypher" not in aggregates or "graph_rag" not in aggregates:
        out.append("*(both approaches required for comparison)*\n")
        return "\n".join(out)

    a = aggregates["text_to_cypher"]
    b = aggregates["graph_rag"]

    findings = []

    # Answered rate comparison
    if a["answered_rate"] > b["answered_rate"]:
        findings.append(
            f"Approach A had a higher answered rate "
            f"({fmt_pct(a['answered_rate'])} vs {fmt_pct(b['answered_rate'])}), "
            f"meaning B occasionally returned errors A did not."
        )
    elif b["answered_rate"] > a["answered_rate"]:
        findings.append(
            f"Approach B had a higher answered rate "
            f"({fmt_pct(b['answered_rate'])} vs {fmt_pct(a['answered_rate'])}), "
            f"reflecting A's known difficulty generating valid aggregation Cypher."
        )

    # Keyword coverage
    if abs(a["mean_keyword_coverage"] - b["mean_keyword_coverage"]) < 0.05:
        findings.append(
            f"Mean keyword coverage was comparable across approaches "
            f"({fmt_pct(a['mean_keyword_coverage'])} vs {fmt_pct(b['mean_keyword_coverage'])})."
        )
    elif a["mean_keyword_coverage"] > b["mean_keyword_coverage"]:
        findings.append(
            f"Approach A had higher mean keyword coverage "
            f"({fmt_pct(a['mean_keyword_coverage'])} vs {fmt_pct(b['mean_keyword_coverage'])})."
        )
    else:
        findings.append(
            f"Approach B had higher mean keyword coverage "
            f"({fmt_pct(b['mean_keyword_coverage'])} vs {fmt_pct(a['mean_keyword_coverage'])})."
        )

    # Latency
    findings.append(
        f"Mean latency: A {fmt_seconds(a['mean_latency_ms'])} vs "
        f"B {fmt_seconds(b['mean_latency_ms'])}."
    )

    # Per-category divergence
    cat_diffs = []
    for cat in a.get("per_category", {}):
        if cat not in b.get("per_category", {}):
            continue
        a_cov = a["per_category"][cat]["mean_keyword_coverage"]
        b_cov = b["per_category"][cat]["mean_keyword_coverage"]
        diff = b_cov - a_cov
        if abs(diff) >= 0.2:
            winner = "B" if diff > 0 else "A"
            cat_diffs.append(
                f"  - **{cat}:** {winner} wins by "
                f"{fmt_pct(abs(diff))} (A={fmt_pct(a_cov)}, B={fmt_pct(b_cov)})"
            )
    if cat_diffs:
        findings.append("Category-level divergence (>20% gap):\n" + "\n".join(cat_diffs))

    out.extend(f"- {f}" for f in findings)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Phase 4 benchmark results"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to results JSON (default: latest evaluation/results_*.json)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write the Markdown report (default: stdout)",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Include full per-question answers in the report",
    )
    args = parser.parse_args()

    # Find input
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_results()
        if not input_path:
            print("No results files found in evaluation/. Run run_benchmark.py first.",
                  file=sys.stderr)
            sys.exit(1)

    if not input_path.exists():
        print(f"Results file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build the report
    sections = [
        "# Phase 4 Benchmark Report\n",
        f"_Source: `{input_path}`_\n",
        section_metadata(data.get("metadata", {})),
        section_overall_summary(data.get("aggregates", {})),
        section_per_category(data.get("aggregates", {})),
        section_per_question(
            data.get("questions", []),
            data.get("scores", []),
            data.get("answers", {}),
        ),
        section_takeaways(data.get("aggregates", {})),
    ]
    if args.detail:
        sections.append(section_detailed_answers(
            data.get("questions", []),
            data.get("answers", {}),
            data.get("scores", []),
        ))

    report = "\n".join(sections)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
