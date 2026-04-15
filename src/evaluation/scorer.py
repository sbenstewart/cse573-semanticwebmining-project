"""Automated scoring for Phase 4 benchmark.

Scoring methodology:

1. **Keyword coverage** — fraction of expected_keywords that appear
   (case-insensitive substring) in the answer text. The primary
   automated metric for correctness.

2. **Citation F1** — if a question has expected_doc_ids, we measure
   precision, recall, and F1 of the cited_doc_ids against the gold.
   Only computed when gold citations are provided.

3. **Latency** — raw wall-clock time in ms. Lower is better.

4. **Error rate** — fraction of answers that returned an error.
   A high error rate is itself a quality signal (Approach A's
   known Cypher aggregation failures show up here).

5. **Answered rate** — fraction of non-error answers. The complement
   of error rate, but useful to report separately.

We deliberately keep scoring simple and interpretable — the point of
Phase 4 is to quantify the tradeoffs between A and B we already see
qualitatively, not to invent new metrics. An LLM-as-judge step could
be added later for more nuanced qualitative scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class QuestionScore:
    """Per-question scoring breakdown."""
    question_id: str
    category: str
    approach: str
    answered: bool                       # False if Answer.error was set
    keyword_coverage: float              # in [0.0, 1.0]
    keywords_matched: list[str]
    keywords_missed: list[str]
    citation_precision: float | None     # None if no gold citations
    citation_recall: float | None
    citation_f1: float | None
    latency_ms: float
    answer_length_chars: int
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "approach": self.approach,
            "answered": self.answered,
            "keyword_coverage": self.keyword_coverage,
            "keywords_matched": self.keywords_matched,
            "keywords_missed": self.keywords_missed,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_f1": self.citation_f1,
            "latency_ms": self.latency_ms,
            "answer_length_chars": self.answer_length_chars,
            "error": self.error,
        }


def score_keywords(
    answer_text: str, expected_keywords: Iterable[str]
) -> tuple[float, list[str], list[str]]:
    """Case-insensitive substring matching for expected keywords.

    Returns (coverage, matched, missed). Coverage is
    len(matched) / len(expected_keywords), or 1.0 if no keywords
    expected (loose/qualitative questions).
    """
    keywords = list(expected_keywords or [])
    if not keywords:
        # No keywords means "qualitative only" — treat as full credit
        # for the automated metric; final judgment is manual.
        return 1.0, [], []

    lower = (answer_text or "").lower()
    matched, missed = [], []
    for kw in keywords:
        if kw.lower() in lower:
            matched.append(kw)
        else:
            missed.append(kw)
    coverage = len(matched) / len(keywords)
    return coverage, matched, missed


def score_citations(
    cited: Iterable[str], gold: Iterable[str] | None
) -> tuple[float | None, float | None, float | None]:
    """Precision / recall / F1 on cited document IDs.

    Returns (None, None, None) when no gold citations are provided.
    """
    if gold is None:
        return None, None, None

    cited_set = set(cited or [])
    gold_set = set(gold)

    if not gold_set and not cited_set:
        return 1.0, 1.0, 1.0
    if not cited_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        # Cited something but gold says nothing expected — precision is
        # undefined in the strict sense; we report 0 for precision and
        # full recall.
        return 0.0, 1.0, 0.0

    true_pos = len(cited_set & gold_set)
    precision = true_pos / len(cited_set)
    recall = true_pos / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_answer(
    question: dict,
    answer,  # common.Answer
) -> QuestionScore:
    """Score a single answer against a single gold question."""
    coverage, matched, missed = score_keywords(
        answer.text, question.get("expected_keywords", [])
    )
    precision, recall, f1 = score_citations(
        answer.cited_doc_ids, question.get("expected_doc_ids")
    )
    return QuestionScore(
        question_id=question["id"],
        category=question["category"],
        approach=answer.approach,
        answered=not answer.is_error(),
        keyword_coverage=coverage,
        keywords_matched=matched,
        keywords_missed=missed,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=f1,
        latency_ms=answer.latency_ms,
        answer_length_chars=len(answer.text or ""),
        error=answer.error,
    )


# ---------------------------------------------------------------------------
# Aggregate statistics across many scores
# ---------------------------------------------------------------------------

@dataclass
class AggregateStats:
    """Summary of many QuestionScores for one approach."""
    approach: str
    n_questions: int
    n_answered: int
    answered_rate: float
    mean_keyword_coverage: float
    mean_latency_ms: float
    median_latency_ms: float
    mean_answer_length: float
    per_category: dict  # category -> {n, keyword_coverage, answered_rate}

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "n_questions": self.n_questions,
            "n_answered": self.n_answered,
            "answered_rate": self.answered_rate,
            "mean_keyword_coverage": self.mean_keyword_coverage,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "mean_answer_length": self.mean_answer_length,
            "per_category": self.per_category,
        }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_scores(
    scores: list[QuestionScore], approach: str
) -> AggregateStats:
    """Compute aggregate statistics for one approach across all questions."""
    by_approach = [s for s in scores if s.approach == approach]
    n = len(by_approach)
    if n == 0:
        return AggregateStats(
            approach=approach, n_questions=0, n_answered=0,
            answered_rate=0.0, mean_keyword_coverage=0.0,
            mean_latency_ms=0.0, median_latency_ms=0.0,
            mean_answer_length=0.0, per_category={},
        )

    answered = [s for s in by_approach if s.answered]
    latencies = [s.latency_ms for s in by_approach]
    coverages = [s.keyword_coverage for s in by_approach]

    # Per-category breakdown
    categories = {s.category for s in by_approach}
    per_category = {}
    for cat in sorted(categories):
        cat_scores = [s for s in by_approach if s.category == cat]
        cat_answered = [s for s in cat_scores if s.answered]
        per_category[cat] = {
            "n": len(cat_scores),
            "n_answered": len(cat_answered),
            "answered_rate": len(cat_answered) / len(cat_scores),
            "mean_keyword_coverage": _mean(
                [s.keyword_coverage for s in cat_scores]
            ),
            "mean_latency_ms": _mean([s.latency_ms for s in cat_scores]),
        }

    return AggregateStats(
        approach=approach,
        n_questions=n,
        n_answered=len(answered),
        answered_rate=len(answered) / n,
        mean_keyword_coverage=_mean(coverages),
        mean_latency_ms=_mean(latencies),
        median_latency_ms=_median(latencies),
        mean_answer_length=_mean(
            [s.answer_length_chars for s in by_approach]
        ),
        per_category=per_category,
    )
