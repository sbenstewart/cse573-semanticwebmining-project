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

STOP_WORDS = {
    "the", "a", "an", "is", "it", "its", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "but", "not", "this", "that",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "can", "from", "by", "as", "up", "so", "if", "than", "then",
    "he", "she", "they", "we", "you", "i", "their", "our", "your",
}


def _tokenize(text: str) -> list:
    tokens = re.findall(r"\b[a-z][a-z0-9]{1,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class BM25Retriever:

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.docs = []
        self.tokenized_corpus = []
        self._is_fitted = False

    def index(self, docs: list) -> None:
        self.docs = docs
        self.tokenized_corpus = [
            _tokenize(d.get("cleaned_text") or d.get("raw_text", ""))
            for d in docs
        ]
        logger.info(f"[BM25] Building index for {len(docs)} documents...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self._is_fitted = True
        logger.info(f"[BM25] Index built.")

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list:
        if not self._is_fitted:
            raise RuntimeError("Index not built. Call index() first.")
        query_tokens = _tokenize(query)
        if not query_tokens:
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
                "snippet": self._snippet(doc.get("cleaned_text") or doc.get("raw_text", ""), query),
                "source_url": doc.get("source_url", ""),
                "publisher": doc.get("publisher", ""),
                "published_date": doc.get("published_date", ""),
                "retrieval_method": "bm25",
            })
        return results

    def save(self, path: Path = INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "docs": self.docs, "tokenized_corpus": self.tokenized_corpus}, f)
        logger.info(f"[BM25] Index saved to {path}")

    def load(self, path: Path = INDEX_PATH) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.docs = data["docs"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self._is_fitted = True
        logger.info(f"[BM25] Index loaded from {path} ({len(self.docs)} docs)")

    def precision_at_k(self, query: str, relevant_doc_ids: set, k: int = TOP_K_RESULTS) -> float:
        results = self.search(query, top_k=k)
        retrieved_ids = {r["doc_id"] for r in results}
        hits = len(retrieved_ids & relevant_doc_ids)
        return hits / k if k > 0 else 0.0

    def ndcg_at_k(self, query: str, relevance_grades: dict, k: int = TOP_K_RESULTS) -> float:
        results = self.search(query, top_k=k)
        dcg = sum(
            (2 ** relevance_grades.get(r["doc_id"], 0) - 1) / math.log2(rank + 1)
            for rank, r in enumerate(results[:k], start=1)
        )
        ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(rank + 1)
            for rank, rel in enumerate(ideal_rels, start=1)
        )
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _snippet(text: str, query: str, window: int = 200) -> str:
        terms = query.lower().split()
        text_lower = text.lower()
        best_pos = next((text_lower.find(t) for t in terms if text_lower.find(t) >= 0), 0)
        start = max(0, best_pos - 50)
        end = min(len(text), best_pos + window)
        return ("..." if start > 0 else "") + text[start:end].strip() + "..."
