"""Thin wrapper around sentence-transformers for document embedding.

Uses BAAI/bge-base-en-v1.5 by default (768 dims, 109M params).
Model is downloaded once on first use (~400 MB) and cached in
~/.cache/huggingface. Fully offline after first run.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default model — good quality/speed tradeoff for a 295-doc corpus.
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768  # bge-base output dimension


class Embedder:
    """Compute dense vector embeddings for text.

    Usage::

        emb = Embedder()                     # loads model on first call
        vecs = emb.embed(["hello", "world"]) # shape (2, 768)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load(self):
        if self._model is not None:
            return
        logger.info(f"[Embedder] Loading {self.model_name} ...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        logger.info(f"[Embedder] Model loaded (dim={self.dim})")

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return EMBEDDING_DIM

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts. Returns shape (len(texts), dim)."""
        self._load()
        # bge models recommend prepending "Represent this sentence: "
        # for retrieval tasks, but empirically it makes little difference
        # at this corpus scale. We skip it for simplicity.
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # unit-norm for cosine similarity
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text. Returns a plain Python list (for Neo4j)."""
        vec = self.embed([text])[0]
        return vec.tolist()
