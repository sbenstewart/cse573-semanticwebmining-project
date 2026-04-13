"""Tests for the KG funding extractor and normalizer.

These tests DO NOT hit a live Ollama; they mock ollama.chat so the tests
run in CI without a local model. The goal is to exercise the JSON parsing,
validation, and normalization logic.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.kg.extractor import FundingExtractor, FundingRound
from src.kg.normalizer import (
    canonical_investor, canonical_startup, normalize_name, INVESTOR_ALIASES
)


# --- normalizer -----------------------------------------------------------

class TestNormalizeName:
    def test_lowercases(self):
        assert normalize_name("Replit") == "replit"

    def test_strips_punctuation(self):
        assert normalize_name("Acme, Inc.") == "acme"

    def test_strips_legal_suffix_llc(self):
        assert normalize_name("Foo LLC") == "foo"

    def test_strips_legal_suffix_inc(self):
        assert normalize_name("Foo Inc") == "foo"

    def test_collapses_whitespace(self):
        assert normalize_name("  foo   bar  ") == "foo bar"

    def test_empty_input(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_unicode_folding(self):
        # full-width "A" (U+FF21) should fold to ASCII
        assert normalize_name("\uFF21CME") == "acme"


class TestCanonicalInvestor:
    def test_a16z_aliases_to_andreessen(self):
        display, norm = canonical_investor("a16z")
        assert display == "Andreessen Horowitz"
        assert norm == "andreessen horowitz"

    def test_a16z_uppercase(self):
        display, norm = canonical_investor("A16Z")
        assert display == "Andreessen Horowitz"

    def test_yc_aliases_to_y_combinator(self):
        display, norm = canonical_investor("YC")
        assert display == "Y Combinator"

    def test_unknown_investor_kept_as_is(self):
        display, norm = canonical_investor("Acme Ventures")
        assert display == "Acme Ventures"
        assert norm == "acme ventures"

    def test_alias_table_is_consistent(self):
        # Every alias should normalize to itself or to a known canonical form
        for key in INVESTOR_ALIASES:
            assert normalize_name(key) == key, f"Alias key not pre-normalized: {key}"


class TestCanonicalStartup:
    def test_returns_both_forms(self):
        display, norm = canonical_startup("Replit")
        assert display == "Replit"
        assert norm == "replit"


# --- extractor ------------------------------------------------------------

def _mock_ollama_response(payload: dict):
    """Helper: build a fake ollama.chat response around a JSON payload."""
    return {"message": {"content": json.dumps(payload)}}


REPLIT_DOC = {
    "doc_id": "doc-replit-1",
    "title": "Replit grabs $400M at $9B valuation",
    "cleaned_text": (
        "Replit raised $400M at a $9B valuation led by a16z with participation "
        "from Founders Fund and Khosla Ventures, announced January 2026."
    ),
}

OSKA_DOC = {
    "doc_id": "doc-oska-1",
    "title": "Oska Health raises 11M seed",
    "cleaned_text": "Oska Health raised a 11 million seed round in March 2026.",
}

NOT_A_FUNDING_DOC = {
    "doc_id": "doc-asus-1",
    "title": "ASUS Helps Scale AI-Driven Operations",
    "cleaned_text": "ASUS announced a new edge computing platform for industrial AI.",
}


class TestFundingExtractor:
    def test_parses_valid_json(self):
        ext = FundingExtractor()
        payload = {"rounds": [{
            "company": "Replit",
            "amount_raw": "$400M",
            "amount_usd": 400000000,
            "round_type": "Series C",
            "valuation_raw": "$9B",
            "valuation_usd": 9000000000,
            "announced_date": "2026-01",
            "investors": ["a16z", "Founders Fund", "Khosla Ventures"],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert len(rounds) == 1
        r = rounds[0]
        assert r.company == "Replit"
        assert r.amount_usd == 400000000
        assert r.round_type == "Series C"
        assert r.investors == ["a16z", "Founders Fund", "Khosla Ventures"]
        assert r.source_doc_id == "doc-replit-1"
        assert len(r.round_id) == 12  # sha1 prefix

    def test_empty_rounds_for_non_funding_doc(self):
        ext = FundingExtractor()
        payload = {"rounds": []}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(NOT_A_FUNDING_DOC)
        assert rounds == []

    def test_skips_round_with_no_amount(self):
        ext = FundingExtractor()
        payload = {"rounds": [{
            "company": "MysteryCo",
            "amount_raw": "undisclosed",
            "amount_usd": None,
            "round_type": "Series B",
            "investors": ["Some VC"],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert rounds == []

    def test_skips_round_with_no_company(self):
        ext = FundingExtractor()
        payload = {"rounds": [{
            "company": "",
            "amount_usd": 5000000,
            "investors": [],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert rounds == []

    def test_coerces_string_amount_to_int(self):
        ext = FundingExtractor()
        payload = {"rounds": [{
            "company": "Foo",
            "amount_raw": "$5M",
            "amount_usd": "5000000",  # string, not int
            "round_type": "Seed",
            "investors": [],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert len(rounds) == 1
        assert rounds[0].amount_usd == 5000000

    def test_retries_on_malformed_json(self):
        ext = FundingExtractor()
        bad = {"message": {"content": "not json at all"}}
        good = _mock_ollama_response({"rounds": []})
        with patch("ollama.chat", side_effect=[bad, good]) as m:
            rounds = ext.extract(REPLIT_DOC)
        assert rounds == []
        assert m.call_count == 2  # one retry happened

    def test_salvages_json_wrapped_in_prose(self):
        ext = FundingExtractor()
        wrapped = {"message": {"content":
            'Sure! Here is the JSON:\n{"rounds": []}\nHope that helps.'
        }}
        with patch("ollama.chat", return_value=wrapped):
            rounds = ext.extract(REPLIT_DOC)
        assert rounds == []


    def test_drops_aggregate_attribution(self):
        """Two companies, same amount, same year → both dropped."""
        ext = FundingExtractor()
        payload = {"rounds": [
            {"company": "Anthropic", "amount_raw": "$1.5B", "amount_usd": 1500000000,
             "round_type": "Series C", "announced_date": "2023-02", "investors": []},
            {"company": "Higgsfield AI", "amount_raw": "$1.5B", "amount_usd": 1500000000,
             "round_type": "Series C", "announced_date": "2023-02", "investors": []},
        ]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert rounds == [], "aggregate-attribution duplicates should be dropped"

    def test_keeps_distinct_amounts_for_different_companies(self):
        """Different amounts → both kept (real co-occurrence in a roundup)."""
        ext = FundingExtractor()
        payload = {"rounds": [
            {"company": "Replit", "amount_raw": "$400M", "amount_usd": 400000000,
             "round_type": "Series C", "announced_date": "2026-01", "investors": []},
            {"company": "Coherent", "amount_raw": "$2B", "amount_usd": 2000000000,
             "round_type": "Growth", "announced_date": "2025-12", "investors": []},
        ]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert len(rounds) == 2

    def test_keeps_same_company_multiple_rounds(self):
        """Same company, two distinct rounds → both kept."""
        ext = FundingExtractor()
        payload = {"rounds": [
            {"company": "Replit", "amount_raw": "$200M", "amount_usd": 200000000,
             "round_type": "Series C", "announced_date": "2022-06", "investors": []},
            {"company": "Replit", "amount_raw": "$400M", "amount_usd": 400000000,
             "round_type": "Growth", "announced_date": "2026-01", "investors": []},
        ]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            rounds = ext.extract(REPLIT_DOC)
        assert len(rounds) == 2


class TestFundingRoundId:
    def test_deterministic(self):
        a = FundingRound(company="Replit", amount_raw="$400M", amount_usd=400000000,
                         round_type="Series C", valuation_raw=None, valuation_usd=None,
                         announced_date="2026-01", investors=[])
        b = FundingRound(company="Replit", amount_raw="$400M", amount_usd=400000000,
                         round_type="Series C", valuation_raw=None, valuation_usd=None,
                         announced_date="2026-01-15", investors=[])
        # Same company + amount + year → same ID (day-level variance ignored)
        assert a.compute_round_id() == b.compute_round_id()

    def test_different_amount_produces_different_id(self):
        a = FundingRound(company="Replit", amount_raw="$400M", amount_usd=400000000,
                         round_type="Series C", valuation_raw=None, valuation_usd=None,
                         announced_date="2026-01", investors=[])
        b = FundingRound(company="Replit", amount_raw="$500M", amount_usd=500000000,
                         round_type="Series C", valuation_raw=None, valuation_usd=None,
                         announced_date="2026-01", investors=[])
        assert a.compute_round_id() != b.compute_round_id()


# --- ProductTechExtractor (Pass 2) ---------------------------------------

from src.kg.extractor import ProductTechExtractor, ProductMention


class TestProductTechExtractor:
    DOC = {"doc_id": "doc-1", "title": "T", "cleaned_text": "ignored, mocked"}

    def test_extracts_named_product_with_tech(self):
        ext = ProductTechExtractor()
        payload = {"items": [{
            "company": "Anthropic",
            "product": "Claude 3.5 Sonnet",
            "description": "A frontier language model",
            "technologies": ["transformer", "RLHF"],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            mentions = ext.extract(self.DOC)
        assert len(mentions) == 1
        m = mentions[0]
        assert m.company == "Anthropic"
        assert m.product == "Claude 3.5 Sonnet"
        assert m.technologies == ["transformer", "RLHF"]

    def test_drops_generic_product_name(self):
        ext = ProductTechExtractor()
        payload = {"items": [{
            "company": "OpenAI",
            "product": "AI platform",  # generic, blocked
            "description": None,
            "technologies": [],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            mentions = ext.extract(self.DOC)
        assert mentions == []

    def test_drops_product_equal_to_company(self):
        ext = ProductTechExtractor()
        payload = {"items": [{
            "company": "Replit", "product": "Replit",
            "description": None, "technologies": [],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            mentions = ext.extract(self.DOC)
        assert mentions == []

    def test_filters_generic_techs(self):
        ext = ProductTechExtractor()
        payload = {"items": [{
            "company": "Acme",
            "product": "Acme Coder Pro",
            "description": None,
            "technologies": ["LangChain", "AI", "vector database", "ML"],
        }]}
        with patch("ollama.chat", return_value=_mock_ollama_response(payload)):
            mentions = ext.extract(self.DOC)
        assert len(mentions) == 1
        assert "LangChain" in mentions[0].technologies
        assert "vector database" in mentions[0].technologies
        assert "AI" not in mentions[0].technologies
        assert "ML" not in mentions[0].technologies

    def test_empty_items_for_non_product_doc(self):
        ext = ProductTechExtractor()
        with patch("ollama.chat", return_value=_mock_ollama_response({"items": []})):
            mentions = ext.extract(self.DOC)
        assert mentions == []
