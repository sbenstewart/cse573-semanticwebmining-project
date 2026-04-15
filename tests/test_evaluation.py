"""Tests for Phase 4 scorer."""
from __future__ import annotations

import pytest

from src.evaluation.scorer import (
    AggregateStats,
    QuestionScore,
    aggregate_scores,
    score_answer,
    score_citations,
    score_keywords,
)
from src.rag.common import Answer


# ---------------------------------------------------------------------------
# score_keywords
# ---------------------------------------------------------------------------

class TestScoreKeywords:
    def test_all_present(self):
        cov, matched, missed = score_keywords(
            "Founders Fund, GV, and Benchmark invested in Replit.",
            ["Founders Fund", "GV", "Benchmark"],
        )
        assert cov == 1.0
        assert set(matched) == {"Founders Fund", "GV", "Benchmark"}
        assert missed == []

    def test_partial_match(self):
        cov, matched, missed = score_keywords(
            "Founders Fund invested.",
            ["Founders Fund", "GV", "Benchmark"],
        )
        assert cov == pytest.approx(1 / 3)
        assert matched == ["Founders Fund"]
        assert set(missed) == {"GV", "Benchmark"}

    def test_case_insensitive(self):
        cov, _, _ = score_keywords("FOUNDERS FUND", ["founders fund"])
        assert cov == 1.0

    def test_no_keywords_gives_full_credit(self):
        cov, matched, missed = score_keywords("anything", [])
        assert cov == 1.0
        assert matched == []
        assert missed == []

    def test_empty_answer(self):
        cov, _, missed = score_keywords("", ["Replit"])
        assert cov == 0.0
        assert missed == ["Replit"]

    def test_none_answer_handled(self):
        cov, _, _ = score_keywords(None, ["Replit"])
        assert cov == 0.0


# ---------------------------------------------------------------------------
# score_citations
# ---------------------------------------------------------------------------

class TestScoreCitations:
    def test_no_gold_returns_none(self):
        assert score_citations(["a", "b"], None) == (None, None, None)

    def test_perfect_match(self):
        p, r, f1 = score_citations(["a", "b"], ["a", "b"])
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_partial_match(self):
        # cited {a, b}, gold {a, c} → tp=1, precision=0.5, recall=0.5, f1=0.5
        p, r, f1 = score_citations(["a", "b"], ["a", "c"])
        assert p == 0.5
        assert r == 0.5
        assert f1 == 0.5

    def test_empty_cited(self):
        p, r, f1 = score_citations([], ["a"])
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_both_empty(self):
        p, r, f1 = score_citations([], [])
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_f1_formula(self):
        # 2 cited, 4 gold, 1 tp → p=0.5, r=0.25, f1=2*0.5*0.25/0.75 = 1/3
        p, r, f1 = score_citations(["a", "b"], ["a", "c", "d", "e"])
        assert p == 0.5
        assert r == 0.25
        assert f1 == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# score_answer
# ---------------------------------------------------------------------------

class TestScoreAnswer:
    def test_successful_answer(self):
        question = {
            "id": "F1",
            "category": "factual_lookup",
            "expected_keywords": ["Replit", "Founders Fund"],
            "expected_doc_ids": ["doc1", "doc2"],
        }
        ans = Answer(
            question="Who invested in Replit?",
            text="Founders Fund invested in Replit.",
            cited_doc_ids=["doc1", "doc3"],
            approach="text_to_cypher",
            latency_ms=150.0,
        )
        score = score_answer(question, ans)
        assert score.question_id == "F1"
        assert score.category == "factual_lookup"
        assert score.answered
        assert score.keyword_coverage == 1.0
        assert score.citation_precision == 0.5
        assert score.citation_recall == 0.5
        assert score.latency_ms == 150.0

    def test_error_answer(self):
        question = {"id": "A1", "category": "aggregation",
                    "expected_keywords": ["OpenAI"]}
        ans = Answer(
            question="q", text="",
            approach="text_to_cypher",
            latency_ms=5000.0,
            error="CypherSyntaxError",
        )
        score = score_answer(question, ans)
        assert not score.answered
        assert score.keyword_coverage == 0.0
        assert score.error == "CypherSyntaxError"

    def test_no_citations_in_gold(self):
        question = {"id": "X", "category": "semantic",
                    "expected_keywords": ["agent"]}
        ans = Answer(question="q", text="Replit Agent mentioned.",
                     cited_doc_ids=["d1"], approach="graph_rag",
                     latency_ms=10.0)
        score = score_answer(question, ans)
        assert score.citation_precision is None
        assert score.citation_f1 is None
        assert score.keyword_coverage == 1.0


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------

class TestAggregateScores:
    def _mk(self, **kwargs):
        defaults = dict(
            question_id="q", category="factual_lookup",
            approach="text_to_cypher", answered=True,
            keyword_coverage=1.0, keywords_matched=[], keywords_missed=[],
            citation_precision=None, citation_recall=None, citation_f1=None,
            latency_ms=100.0, answer_length_chars=50,
        )
        defaults.update(kwargs)
        return QuestionScore(**defaults)

    def test_empty_scores(self):
        agg = aggregate_scores([], "text_to_cypher")
        assert agg.n_questions == 0
        assert agg.answered_rate == 0.0

    def test_filters_by_approach(self):
        scores = [
            self._mk(approach="text_to_cypher", keyword_coverage=1.0),
            self._mk(approach="graph_rag", keyword_coverage=0.5),
        ]
        agg = aggregate_scores(scores, "text_to_cypher")
        assert agg.n_questions == 1
        assert agg.mean_keyword_coverage == 1.0

    def test_mean_and_median_latency(self):
        scores = [
            self._mk(latency_ms=100),
            self._mk(latency_ms=200),
            self._mk(latency_ms=300),
        ]
        agg = aggregate_scores(scores, "text_to_cypher")
        assert agg.mean_latency_ms == 200.0
        assert agg.median_latency_ms == 200.0

    def test_answered_rate(self):
        scores = [
            self._mk(answered=True),
            self._mk(answered=False, error="boom"),
            self._mk(answered=True),
        ]
        agg = aggregate_scores(scores, "text_to_cypher")
        assert agg.n_answered == 2
        assert agg.answered_rate == pytest.approx(2 / 3)

    def test_per_category_breakdown(self):
        scores = [
            self._mk(category="factual_lookup", keyword_coverage=1.0),
            self._mk(category="factual_lookup", keyword_coverage=0.5),
            self._mk(category="aggregation", keyword_coverage=0.0),
        ]
        agg = aggregate_scores(scores, "text_to_cypher")
        assert "factual_lookup" in agg.per_category
        assert "aggregation" in agg.per_category
        assert agg.per_category["factual_lookup"]["n"] == 2
        assert agg.per_category["factual_lookup"]["mean_keyword_coverage"] == 0.75
        assert agg.per_category["aggregation"]["mean_keyword_coverage"] == 0.0
