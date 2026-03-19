"""
src/retrieval/bm25_retriever.py

BM25 (Okapi BM25) retriever — Phase 1 classical baseline.

BM25 is the dominant probabilistic retrieval function used in
production search systems (Elasticsearch, Solr, Lucene).
It improves on TF-IDF with:
  - Term frequency saturation (k1 parameter)
  - Document length normalization (b parameter)

Reference:
  Robertson & Zaragoza (2009) — "The Probabilistic Relevance Framework:
  BM25 and Beyond". Foundations and Trends in Information Retrieval.

Implementation:
  Uses `rank-bm25` library (pip install rank-bm25).
"""

import re
import math
import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BM25_K1, BM25_B, TOP_K_RESULTS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

INDEX_PATH = PROCESSED_DATA_DIR / "bm25_index.pkl"

# Basic English stop words
STOP_WORDS = {
    "the", "a", "an", "is", "it", "its", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "but", "not", "this", "that",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "can", "from", "by", "as", "up", "so", "if", "than", "then",
    "he", "she", "they", "we", "you", "i", "their", "our", "your",
}


def _tokenize(text: str) -> list[str]:
    """
    Lowercase tokenization with stop word removal.
    BM25Okapi expects pre-tokenized input.
    """
    tokens = re.findall(r"\b[a-z][a-z0-9]{1,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class BM25Retriever:
    """
    BM25 (Okapi) retriever over the TrendScout corpus.

    Usage:
        retriever = BM25Retriever()
        retriever.index(docs)
        results = retriever.search("LLM startup funding 2024", top_k=5)
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.docs: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []
        self._is_fitted = False

    # ── Public Interface ─────────────────────────────────────

    def index(self, docs: list[dict]) -> None:
        """Build BM25 index from a list of document dicts."""
        self.docs = docs
        self.tokenized_corpus = [
            _tokenize(d.get("cleaned_text") or d.get("raw_text", ""))
            for d in docs
        ]

        logger.info(f"[BM25] Building index for {len(docs)} documents...")
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        self._is_fitted = True

        avg_dl = sum(len(t) for t in self.tokenized_corpus) / max(len(self.tokenized_corpus), 1)
        logger.info(f"[BM25] Index built. Avg doc length: {avg_dl:.1f} tokens")

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Retrieve top-K documents by BM25 score.

        Returns:
            List of dicts with keys: doc_id, title, score, snippet, source_url
        """
        if not self._is_fitted:
            raise RuntimeError("Index not built. Call index() first.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("[BM25] Empty query after tokenization")
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc = self.docs[idx]
            results.append({
                "doc_id": doc.get("doc_id", f"doc_{idx}"),
                "title": doc.get("title", "(no title)"),
                "score": float(scores[idx]),
                "snippet": self._snippet(
                    doc.get("cleaned_text") or doc.get("raw_text", ""), query
                ),
                "source_url": doc.get("source_url", ""),
                "publisher": doc.get("publisher", ""),
                "published_date": doc.get("published_date", ""),
                "retrieval_method": "bm25",
            })

        return results

    def save(self, path: Path = INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "docs": self.docs,
                "tokenized_corpus": self.tokenized_corpus,
                "k1": self.k1,
                "b": self.b,
            }, f)
        logger.info(f"[BM25] Index saved to {path}")

    def load(self, path: Path = INDEX_PATH) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.docs = data["docs"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self._is_fitted = True
        logger.info(f"[BM25] Index loaded from {path} ({len(self.docs)} docs)")

    # ── Evaluation Helpers ───────────────────────────────────

    def precision_at_k(
        self,
        query: str,
        relevant_doc_ids: set[str],
        k: int = TOP_K_RESULTS,
    ) -> float:
        """Compute Precision@K given relevant doc IDs."""
        results = self.search(query, top_k=k)
        retrieved_ids = {r["doc_id"] for r in results}
        hits = len(retrieved_ids & relevant_doc_ids)
        return hits / k if k > 0 else 0.0

    def ndcg_at_k(
        self,
        query: str,
        relevance_grades: dict[str, int],
        k: int = TOP_K_RESULTS,
    ) -> float:
        """
        Compute nDCG@K.
        relevance_grades: {doc_id: grade} where grade ∈ {0, 1, 2, 3}
        Reference: Järvelin & Kekäläinen (2002).
        """
        results = self.search(query, top_k=k)

        dcg = 0.0
        for rank, result in enumerate(results[:k], start=1):
            rel = relevance_grades.get(result["doc_id"], 0)
            dcg += (2 ** rel - 1) / math.log2(rank + 1)

        # Ideal DCG
        ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(rank + 1)
            for rank, rel in enumerate(ideal_rels, start=1)
        )

        return dcg / idcg if idcg > 0 else 0.0

    # ── Private ──────────────────────────────────────────────

    @staticmethod
    def _snippet(text: str, query: str, window: int = 200) -> str:
        """Extract a snippet centered on the first query term match."""
        terms = query.lower().split()
        text_lower = text.lower()
        best_pos = next(
            (text_lower.find(t) for t in terms if text_lower.find(t) >= 0),
            0
        )
        start = max(0, best_pos - 50)
        end = min(len(text), best_pos + window)
        prefix = "..." if start > 0 else ""
        return prefix + text[start:end].strip() + "..."
