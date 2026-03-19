import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import TFIDF_MAX_FEATURES, TOP_K_RESULTS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

INDEX_PATH = PROCESSED_DATA_DIR / "tfidf_index.pkl"


class TFIDFRetriever:

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=2,
        )
        self.doc_matrix = None
        self.docs = []
        self._is_fitted = False

    def index(self, docs: list) -> None:
        self.docs = docs
        texts = [d.get("cleaned_text") or d.get("raw_text", "") for d in docs]
        logger.info(f"[TF-IDF] Fitting vectorizer on {len(texts)} documents...")
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        logger.info(f"[TF-IDF] Index built. Matrix shape: {self.doc_matrix.shape}")

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list:
        if not self._is_fitted:
            raise RuntimeError("Index not built. Call index() first.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] == 0:
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
                "retrieval_method": "tfidf",
            })
        return results

    def save(self, path: Path = INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "doc_matrix": self.doc_matrix, "docs": self.docs}, f)
        logger.info(f"[TF-IDF] Index saved to {path}")

    def load(self, path: Path = INDEX_PATH) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.doc_matrix = data["doc_matrix"]
        self.docs = data["docs"]
        self._is_fitted = True
        logger.info(f"[TF-IDF] Index loaded from {path} ({len(self.docs)} docs)")

    def precision_at_k(self, query: str, relevant_doc_ids: set, k: int = TOP_K_RESULTS) -> float:
        results = self.search(query, top_k=k)
        retrieved_ids = {r["doc_id"] for r in results}
        hits = len(retrieved_ids & relevant_doc_ids)
        return hits / k if k > 0 else 0.0

    @staticmethod
    def _snippet(text: str, query: str, window: int = 200) -> str:
        query_terms = query.lower().split()
        text_lower = text.lower()
        best_pos = len(text)
        for term in query_terms:
            pos = text_lower.find(term)
            if 0 <= pos < best_pos:
                best_pos = pos
        if best_pos == len(text):
            return text[:window] + "..."
        start = max(0, best_pos - 50)
        end = min(len(text), best_pos + window)
        return ("..." if start > 0 else "") + text[start:end].strip() + "..."
