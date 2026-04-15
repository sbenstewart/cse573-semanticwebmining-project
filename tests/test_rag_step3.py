"""Tests for Phase 3 Step 3: embedder + vector store."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from src.rag.embedder import Embedder, EMBEDDING_DIM, DEFAULT_MODEL
from src.rag.vector_store import VectorStore, INDEX_NAME


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_default_model_and_dim(self):
        emb = Embedder()
        assert emb.model_name == DEFAULT_MODEL
        assert emb.dim == EMBEDDING_DIM  # before loading, returns constant

    def test_embed_calls_model(self):
        """Mock sentence-transformers to avoid downloading the real model."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 768).astype("float32")
        mock_model.get_sentence_embedding_dimension.return_value = 768

        emb = Embedder()
        emb._model = mock_model  # inject mock

        result = emb.embed(["hello", "world", "test"])
        assert result.shape == (3, 768)
        mock_model.encode.assert_called_once()

    def test_embed_one_returns_list(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 768), dtype="float32")
        mock_model.get_sentence_embedding_dimension.return_value = 768

        emb = Embedder()
        emb._model = mock_model

        result = emb.embed_one("hello")
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_lazy_loading(self):
        """Model should NOT be loaded on construction."""
        emb = Embedder()
        assert emb._model is None


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def _mock_client(self, read_return=None, write_return=None):
        client = MagicMock()
        client.run_read.return_value = read_return or []
        client.run_write.return_value = write_return or []
        return client

    def test_create_index(self):
        client = self._mock_client()
        store = VectorStore(client, dim=768)
        store.create_index()
        client.run_write.assert_called_once()
        cypher = client.run_write.call_args[0][0]
        assert "CREATE VECTOR INDEX" in cypher
        assert "doc_embedding" in cypher
        assert "768" in cypher
        assert "cosine" in cypher

    def test_drop_index(self):
        client = self._mock_client()
        store = VectorStore(client)
        store.drop_index()
        cypher = client.run_write.call_args[0][0]
        assert "DROP INDEX" in cypher

    def test_index_exists_true(self):
        client = self._mock_client(read_return=[{"name": INDEX_NAME}])
        store = VectorStore(client)
        assert store.index_exists()

    def test_index_exists_false(self):
        client = self._mock_client(read_return=[])
        store = VectorStore(client)
        assert not store.index_exists()

    def test_store_embedding(self):
        client = self._mock_client()
        store = VectorStore(client)
        store.store_embedding("doc-123", [0.1, 0.2, 0.3])
        client.run_write.assert_called_once()
        cypher = client.run_write.call_args[0][0]
        assert "SET d.embedding" in cypher
        params = client.run_write.call_args[0][1]
        assert params["doc_id"] == "doc-123"
        assert params["embedding"] == [0.1, 0.2, 0.3]

    def test_store_embeddings_batch(self):
        client = self._mock_client(write_return=[{"n": 3}])
        store = VectorStore(client)
        items = [
            {"doc_id": f"d{i}", "embedding": [float(i)] * 768}
            for i in range(3)
        ]
        n = store.store_embeddings_batch(items, batch_size=10)
        assert n == 3
        client.run_write.assert_called_once()
        cypher = client.run_write.call_args[0][0]
        assert "UNWIND" in cypher

    def test_store_embeddings_batch_multiple_batches(self):
        # 5 items with batch_size=2 → 3 calls
        client = self._mock_client(write_return=[{"n": 2}])
        store = VectorStore(client)
        items = [
            {"doc_id": f"d{i}", "embedding": [0.0]}
            for i in range(5)
        ]
        n = store.store_embeddings_batch(items, batch_size=2)
        assert client.run_write.call_count == 3  # 2+2+1

    def test_query(self):
        client = self._mock_client(read_return=[
            {"doc_id": "d1", "title": "Doc 1", "score": 0.95,
             "publisher": "TC", "url": "http://x", "published_date": "2026-01-01"},
            {"doc_id": "d2", "title": "Doc 2", "score": 0.89,
             "publisher": "VB", "url": "http://y", "published_date": "2026-01-02"},
        ])
        store = VectorStore(client)
        results = store.query([0.1] * 768, k=5)
        assert len(results) == 2
        assert results[0]["doc_id"] == "d1"
        assert results[0]["score"] == 0.95
        # Verify the Cypher used the correct index and params
        cypher = client.run_read.call_args[0][0]
        assert "db.index.vector.queryNodes" in cypher
        params = client.run_read.call_args[0][1]
        assert params["index"] == INDEX_NAME
        assert params["k"] == 5
