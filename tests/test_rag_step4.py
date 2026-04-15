"""Tests for Phase 3 Step 4: GraphRAG pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.common import Answer
from src.rag.graph_rag import GraphRAG, _build_context_block


# ---------------------------------------------------------------------------
# Context block formatting
# ---------------------------------------------------------------------------

class TestBuildContextBlock:
    def test_full_context(self):
        doc = {"title": "Replit raises $400M", "publisher": "TC",
               "published_date": "2026-01-16"}
        gn = {
            "startups": ["Replit"],
            "investors": ["Founders Fund", "GV"],
            "funding_rounds": [
                {"startup": "Replit", "amount": "$400M", "round_type": "Growth"}
            ],
            "products": ["Replit Agent"],
            "technologies": ["agentic AI"],
        }
        block = _build_context_block(doc, "Some article text here", gn)
        assert "Replit raises $400M" in block
        assert "TC" in block
        assert "Founders Fund" in block
        assert "Replit Agent" in block
        assert "agentic AI" in block
        assert "$400M" in block

    def test_empty_neighborhood(self):
        doc = {"title": "Empty doc"}
        block = _build_context_block(doc, "", {})
        assert "Empty doc" in block
        assert "Investors" not in block

    def test_deduplicates_funding_rounds(self):
        gn = {
            "funding_rounds": [
                {"startup": "X", "amount": "$1B", "round_type": "A"},
                {"startup": "X", "amount": "$1B", "round_type": "A"},
            ],
        }
        block = _build_context_block({}, "", gn)
        assert block.count("X: $1B") == 1  # deduped


# ---------------------------------------------------------------------------
# GraphRAG pipeline (fully mocked)
# ---------------------------------------------------------------------------

class TestGraphRAG:
    def _setup(self, *, retrieved=None, neighborhood=None,
               llm_answer="A nice answer."):
        client = MagicMock()
        embedder = MagicMock()
        store = MagicMock()

        embedder.embed_one.return_value = [0.1] * 768

        store.query.return_value = retrieved if retrieved is not None else [
            {"doc_id": "d1", "title": "Doc 1", "score": 0.9,
             "publisher": "TC", "url": "http://x",
             "published_date": "2026-01-01"},
        ]

        # run_read is called for graph expansion
        client.run_read.return_value = neighborhood if neighborhood is not None else [
            {
                "doc_title": "Doc 1",
                "doc_publisher": "TC",
                "doc_date": "2026-01-01",
                "startups": ["Replit"],
                "funding_rounds": [{"startup": "Replit", "amount": "$400M",
                                    "amount_usd": 400000000, "round_type": "Growth"}],
                "investors": ["Founders Fund"],
                "products": ["Replit Agent"],
                "technologies": ["agentic AI"],
            }
        ]

        qa = GraphRAG(
            neo4j_client=client,
            embedder=embedder,
            vector_store=store,
        )
        return qa, client, embedder, store

    def test_happy_path(self):
        qa, client, embedder, store = self._setup()
        with patch("ollama.chat") as m:
            m.return_value = {"message": {"content": "Replit raised $400M."}}
            ans = qa.answer("How much did Replit raise?")

        assert isinstance(ans, Answer)
        assert not ans.is_error()
        assert ans.approach == "graph_rag"
        assert "Replit" in ans.text
        assert ans.cited_doc_ids == ["d1"]
        assert ans.latency_ms >= 0
        assert ans.trace["num_context_blocks"] == 1

        embedder.embed_one.assert_called_once()
        store.query.assert_called_once()

    def test_empty_retrieval(self):
        qa, _, _, _ = self._setup(retrieved=[])
        ans = qa.answer("Something obscure")
        assert not ans.is_error()
        assert "couldn't find" in ans.text.lower()

    def test_embedding_failure(self):
        qa, _, embedder, _ = self._setup()
        embedder.embed_one.side_effect = RuntimeError("model not loaded")
        ans = qa.answer("q")
        assert ans.is_error()
        assert "Embedding failed" in ans.error

    def test_vector_search_failure(self):
        qa, _, _, store = self._setup()
        store.query.side_effect = RuntimeError("index not ready")
        ans = qa.answer("q")
        assert ans.is_error()
        assert "Vector search failed" in ans.error

    def test_llm_failure(self):
        qa, _, _, _ = self._setup()
        with patch("ollama.chat", side_effect=RuntimeError("timeout")):
            ans = qa.answer("q")
        assert ans.is_error()
        assert "LLM call failed" in ans.error

    def test_multiple_docs_retrieved(self):
        retrieved = [
            {"doc_id": f"d{i}", "title": f"Doc {i}", "score": 0.9 - i * 0.1,
             "publisher": "TC", "url": f"http://{i}",
             "published_date": "2026-01-01"}
            for i in range(3)
        ]
        qa, _, _, _ = self._setup(retrieved=retrieved)
        with patch("ollama.chat") as m:
            m.return_value = {"message": {"content": "Found 3 relevant docs."}}
            ans = qa.answer("q")

        assert not ans.is_error()
        assert ans.cited_doc_ids == ["d0", "d1", "d2"]
        assert ans.trace["num_context_blocks"] == 3
