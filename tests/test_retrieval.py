import sys
import math
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.retrieval.tfidf_retriever import TFIDFRetriever


SAMPLE_DOCS = [
    {
        "doc_id": "doc_001",
        "title": "Anthropic Raises $2 Billion from Google",
        "cleaned_text": (
            "Anthropic, the AI safety company, has raised two billion dollars "
            "in a new funding round led by Google. The Series C investment will "
            "accelerate Claude model development and AI alignment research."
        ),
        "source_url": "https://techcrunch.com/anthropic-google",
        "publisher": "TechCrunch",
        "published_date": "2024-03-01",
    },
    {
        "doc_id": "doc_002",
        "title": "Mistral AI Releases Open-Source LLM",
        "cleaned_text": (
            "French startup Mistral AI has released a new open-source large language "
            "model. The model outperforms GPT-3.5 on several benchmarks and is freely "
            "available for commercial use. Mistral raised a seed round last year."
        ),
        "source_url": "https://venturebeat.com/mistral",
        "publisher": "VentureBeat",
        "published_date": "2024-02-15",
    },
    {
        "doc_id": "doc_003",
        "title": "OpenAI Hiring LLM Engineers Aggressively",
        "cleaned_text": (
            "OpenAI is on a major hiring spree for LLM engineers, ML researchers, "
            "and AI infrastructure specialists. The company posted over 50 roles for "
            "machine learning and deep learning positions this quarter."
        ),
        "source_url": "https://techcrunch.com/openai-hiring",
        "publisher": "TechCrunch",
        "published_date": "2024-01-20",
    },
    {
        "doc_id": "doc_004",
        "title": "Vector Databases See Record Investment",
        "cleaned_text": (
            "Vector database companies like Pinecone, Weaviate, and Chroma are seeing "
            "record investment as demand for RAG-based applications grows. Pinecone "
            "closed a $100M Series B to scale its managed vector database service."
        ),
        "source_url": "https://techcrunch.com/vector-db",
        "publisher": "TechCrunch",
        "published_date": "2024-03-10",
    },
    {
        "doc_id": "doc_005",
        "title": "Scale AI Expands RLHF Data Labeling Teams",
        "cleaned_text": (
            "Scale AI is expanding its reinforcement learning from human feedback "
            "operations, hiring thousands of contractors for data labeling tasks. "
            "RLHF has become critical infrastructure for training advanced LLMs."
        ),
        "source_url": "https://venturebeat.com/scale-rlhf",
        "publisher": "VentureBeat",
        "published_date": "2024-02-28",
    },
]


class TestBM25Retriever:

    def setup_method(self):
        self.retriever = BM25Retriever()
        self.retriever.index(SAMPLE_DOCS)

    def test_index_builds_without_error(self):
        assert self.retriever._is_fitted is True
        assert self.retriever.bm25 is not None

    def test_search_returns_list(self):
        results = self.retriever.search("LLM funding")
        assert isinstance(results, list)

    def test_search_returns_at_most_top_k(self):
        results = self.retriever.search("AI startup", top_k=3)
        assert len(results) <= 3

    def test_search_results_have_required_fields(self):
        results = self.retriever.search("funding round")
        for r in results:
            assert "doc_id" in r
            assert "title" in r
            assert "score" in r
            assert "snippet" in r
            assert r["retrieval_method"] == "bm25"

    def test_search_scores_are_non_negative(self):
        results = self.retriever.search("LLM engineers hiring")
        for r in results:
            assert r["score"] >= 0

    def test_search_results_sorted_descending(self):
        results = self.retriever.search("funding investment round")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_funding_retrieves_correct_doc(self):
        results = self.retriever.search("Anthropic Google funding", top_k=5)
        doc_ids = [r["doc_id"] for r in results]
        assert "doc_001" in doc_ids

    def test_search_hiring_retrieves_correct_doc(self):
        results = self.retriever.search("LLM engineer hiring OpenAI", top_k=5)
        doc_ids = [r["doc_id"] for r in results]
        assert "doc_003" in doc_ids

    def test_search_vector_database_retrieves_correct_doc(self):
        results = self.retriever.search("vector database Pinecone investment", top_k=5)
        doc_ids = [r["doc_id"] for r in results]
        assert "doc_004" in doc_ids

    def test_search_empty_query_returns_empty(self):
        results = self.retriever.search("")
        assert results == []

    def test_raises_if_not_fitted(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="Call index\\(\\) first"):
            retriever.search("test")

    def test_precision_at_k_returns_float(self):
        precision = self.retriever.precision_at_k(
            query="Anthropic Google funding billion",
            relevant_doc_ids={"doc_001"},
            k=5,
        )
        assert 0.0 <= precision <= 1.0

    def test_ndcg_at_k_returns_float_in_range(self):
        grades = {"doc_001": 3, "doc_002": 1, "doc_003": 0}
        score = self.retriever.ndcg_at_k(
            query="AI funding round",
            relevance_grades=grades,
            k=5,
        )
        assert 0.0 <= score <= 1.0


class TestTFIDFRetriever:

    def setup_method(self):
        self.retriever = TFIDFRetriever()
        self.retriever.index(SAMPLE_DOCS)

    def test_index_builds_without_error(self):
        assert self.retriever._is_fitted is True
        assert self.retriever.doc_matrix is not None

    def test_search_returns_list(self):
        results = self.retriever.search("LLM funding")
        assert isinstance(results, list)

    def test_search_results_have_retrieval_method_tfidf(self):
        results = self.retriever.search("startup funding")
        for r in results:
            assert r["retrieval_method"] == "tfidf"

    def test_search_results_sorted_descending(self):
        results = self.retriever.search("vector database RAG")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_rlhf_returns_correct_doc(self):
        results = self.retriever.search("RLHF reinforcement learning human feedback", top_k=5)
        doc_ids = [r["doc_id"] for r in results]
        assert "doc_005" in doc_ids

    def test_raises_if_not_fitted(self):
        retriever = TFIDFRetriever()
        with pytest.raises(RuntimeError):
            retriever.search("test")

    def test_snippet_extraction(self):
        text = "The quick brown fox jumped over the lazy dog near the river bank."
        snippet = TFIDFRetriever._snippet(text, "fox river")
        assert "fox" in snippet.lower() or "river" in snippet.lower()


class TestBM25Tokenizer:

    def test_tokenize_lowercases(self):
        tokens = _tokenize("Anthropic Raises Funding")
        assert all(t == t.lower() for t in tokens)

    def test_tokenize_removes_stopwords(self):
        tokens = _tokenize("the quick brown fox")
        assert "the" not in tokens

    def test_tokenize_returns_list_of_strings(self):
        tokens = _tokenize("AI startup raises funding round")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_empty_string(self):
        tokens = _tokenize("")
        assert tokens == []

    def test_tokenize_keeps_ai_terms(self):
        tokens = _tokenize("llm transformer bert pytorch")
        assert "llm" in tokens
        assert "transformer" in tokens
