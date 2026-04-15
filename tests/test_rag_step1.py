"""Tests for Phase 3 Step 1: common types + cypher safety validator."""
from __future__ import annotations

import pytest

from src.rag.common import Answer, SCHEMA_PROMPT, ANSWER_STYLE_GUIDE
from src.rag.cypher_safety import (
    UnsafeCypherError, is_read_only, validate_read_only,
)


# --- Answer dataclass -----------------------------------------------------

class TestAnswer:
    def test_minimal_construction(self):
        a = Answer(question="q", text="t")
        assert a.question == "q"
        assert a.text == "t"
        assert a.cited_doc_ids == []
        assert a.approach == "unknown"
        assert a.latency_ms == 0.0
        assert a.trace == {}
        assert a.error is None
        assert not a.is_error()

    def test_error_state(self):
        a = Answer(question="q", text="", error="Timed out")
        assert a.is_error()

    def test_to_dict_round_trip(self):
        a = Answer(
            question="Who invested in Replit?",
            text="Founders Fund, GV, and Benchmark.",
            cited_doc_ids=["doc-1", "doc-2"],
            approach="text_to_cypher",
            latency_ms=123.4,
            trace={"cypher": "MATCH ..."},
        )
        d = a.to_dict()
        assert d["question"] == "Who invested in Replit?"
        assert d["cited_doc_ids"] == ["doc-1", "doc-2"]
        assert d["approach"] == "text_to_cypher"
        assert d["latency_ms"] == 123.4
        assert d["trace"]["cypher"] == "MATCH ..."
        assert d["error"] is None


# --- Schema prompt content guards -----------------------------------------

class TestSchemaPrompt:
    """Lock in critical invariants of the schema prompt so we get a
    test failure if someone later edits it to drop essential info."""

    def test_mentions_all_node_labels(self):
        for label in ("Startup", "Investor", "FundingRound", "Product",
                      "Technology", "Document"):
            assert label in SCHEMA_PROMPT, f"missing {label} in schema prompt"

    def test_mentions_all_relationship_types(self):
        for rel in ("HAS_FUNDING_ROUND", "INVESTED_BY", "SOURCED_FROM",
                    "ANNOUNCED", "USES_TECH", "MENTIONS"):
            assert rel in SCHEMA_PROMPT, f"missing {rel} in schema prompt"

    def test_mentions_normalized_name_convention(self):
        assert "normalized_name" in SCHEMA_PROMPT

    def test_style_guide_forbids_fabrication(self):
        # The style guide must explicitly forbid fabrication / use of
        # background knowledge. Different wordings are fine; we just
        # make sure SOME anti-fabrication language is present.
        guide_lower = ANSWER_STYLE_GUIDE.lower()
        signals = ["do not invent", "do not speculate", "do not add",
                   "must appear", "only the facts", "only what",
                   "do not fabricate"]
        assert any(s in guide_lower for s in signals), (
            "ANSWER_STYLE_GUIDE should explicitly forbid fabrication/speculation"
        )


# --- Cypher safety validator ----------------------------------------------

class TestCypherSafety:
    # ---- accept valid read queries ----

    def test_simple_match(self):
        validate_read_only("MATCH (s:Startup) RETURN s.name")

    def test_match_with_where_and_limit(self):
        validate_read_only(
            "MATCH (s:Startup)-[:HAS_FUNDING_ROUND]->(r:FundingRound) "
            "WHERE r.amount_usd > 1000000 "
            "RETURN s.name, r.amount_raw ORDER BY r.amount_usd DESC LIMIT 10"
        )

    def test_optional_match_start(self):
        validate_read_only(
            "OPTIONAL MATCH (s:Startup)-[:HAS_FUNDING_ROUND]->(r) RETURN s"
        )

    def test_with_aggregation(self):
        validate_read_only(
            "MATCH (i:Investor)<-[:INVESTED_BY]-(r:FundingRound) "
            "RETURN i.name, count(r) AS n ORDER BY n DESC"
        )

    def test_unwind_read_only(self):
        validate_read_only(
            "UNWIND ['replit', 'anthropic'] AS norm "
            "MATCH (s:Startup {normalized_name: norm}) RETURN s"
        )

    def test_is_read_only_boolean_true(self):
        assert is_read_only("MATCH (n) RETURN n")

    # ---- reject destructive writes ----

    def test_rejects_create(self):
        with pytest.raises(UnsafeCypherError, match="CREATE"):
            validate_read_only("CREATE (n:Startup {name: 'Evil'})")

    def test_rejects_merge(self):
        with pytest.raises(UnsafeCypherError, match="MERGE"):
            validate_read_only("MERGE (s:Startup {normalized_name: 'replit'}) RETURN s")

    def test_rejects_delete_in_match_chain(self):
        # "DETACH DELETE n" trips on DETACH first (both are forbidden)
        with pytest.raises(UnsafeCypherError, match="(DELETE|DETACH)"):
            validate_read_only("MATCH (n) DETACH DELETE n")

    def test_rejects_set(self):
        with pytest.raises(UnsafeCypherError, match="SET"):
            validate_read_only("MATCH (s:Startup {name: 'Replit'}) SET s.evil = true RETURN s")

    def test_rejects_remove(self):
        with pytest.raises(UnsafeCypherError, match="REMOVE"):
            validate_read_only("MATCH (s:Startup) REMOVE s.name RETURN s")

    def test_rejects_call(self):
        with pytest.raises(UnsafeCypherError, match="CALL"):
            validate_read_only("CALL apoc.export.cypher.all('/tmp/evil.cypher', {})")

    def test_rejects_load_csv(self):
        with pytest.raises(UnsafeCypherError, match="LOAD"):
            validate_read_only("LOAD CSV FROM 'http://evil.com/x.csv' AS row RETURN row")

    def test_rejects_drop(self):
        with pytest.raises(UnsafeCypherError, match="DROP"):
            validate_read_only("DROP CONSTRAINT startup_name")

    def test_rejects_empty(self):
        with pytest.raises(UnsafeCypherError):
            validate_read_only("")
        with pytest.raises(UnsafeCypherError):
            validate_read_only("   ")

    def test_rejects_invalid_first_clause(self):
        with pytest.raises(UnsafeCypherError, match="start with"):
            validate_read_only("HELLO WORLD")

    def test_is_read_only_boolean_false(self):
        assert not is_read_only("MATCH (n) DELETE n")

    # ---- literal / comment handling ----

    def test_ignores_forbidden_keyword_inside_string_literal(self):
        """A query that happens to contain the word DELETE inside a
        string literal is fine — that's just data, not an op."""
        validate_read_only(
            "MATCH (d:Document) WHERE d.title CONTAINS 'DELETE' RETURN d.title"
        )

    def test_ignores_forbidden_keyword_inside_double_quoted_literal(self):
        validate_read_only(
            'MATCH (d:Document) WHERE d.title CONTAINS "CREATE" RETURN d.title'
        )

    def test_ignores_forbidden_keyword_in_line_comment(self):
        validate_read_only(
            "MATCH (s:Startup) RETURN s.name // we are NOT going to DELETE"
        )

    def test_ignores_forbidden_keyword_in_block_comment(self):
        validate_read_only(
            "MATCH (s:Startup) /* no DELETE or MERGE happens here */ RETURN s.name"
        )

    def test_keyword_in_column_or_property_name_still_rejected(self):
        """If a property name genuinely is 'CREATE', the raw query still
        fails — we err on the side of safety. If this comes up in
        practice, we'd extend the validator to be AST-aware."""
        with pytest.raises(UnsafeCypherError):
            validate_read_only("MATCH (n) RETURN n.CREATE")
