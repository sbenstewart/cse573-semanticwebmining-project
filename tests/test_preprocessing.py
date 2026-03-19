import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.deduplicator import Deduplicator, _tokenize


class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner()

    def _make_doc(self, text: str, doc_id: str = "test_001") -> dict:
        return {
            "doc_id": doc_id,
            "source_url": "https://example.com",
            "publisher": "Test",
            "raw_text": text,
            "cleaned_text": "",
        }

    def test_clean_removes_boilerplate(self):
        doc = self._make_doc("Anthropic raises $2B in new funding. Subscribe to our newsletter for more." * 5)
        result = self.cleaner.clean(doc)
        assert result is not None
        assert "Subscribe to" not in result["cleaned_text"]

    def test_clean_removes_urls(self):
        doc = self._make_doc("Visit https://techcrunch.com/article/123 for more details about AI funding." * 5)
        result = self.cleaner.clean(doc)
        assert result is not None
        assert "https://" not in result["cleaned_text"]

    def test_clean_normalizes_whitespace(self):
        doc = self._make_doc("AI   startup    raises   funding.   " * 15)
        result = self.cleaner.clean(doc)
        assert result is not None
        assert "  " not in result["cleaned_text"]

    def test_clean_returns_none_for_empty_text(self):
        doc = self._make_doc("")
        result = self.cleaner.clean(doc)
        assert result is None

    def test_clean_returns_none_for_too_short_text(self):
        doc = self._make_doc("Short.")
        result = self.cleaner.clean(doc)
        assert result is None

    def test_clean_returns_none_for_non_english(self):
        doc = self._make_doc("这是一段中文文字，用来测试语言过滤器是否工作正常。" * 10)
        result = self.cleaner.clean(doc)
        assert result is None

    def test_clean_passes_valid_english_doc(self):
        doc = self._make_doc(
            "Anthropic, the AI safety company founded in 2021, has raised $2 billion "
            "in its latest funding round led by Google. The company plans to expand "
            "its research into AI alignment and safety at scale." * 2
        )
        result = self.cleaner.clean(doc)
        assert result is not None
        assert len(result["cleaned_text"]) > 100

    def test_clean_normalizes_iso_date(self):
        doc = self._make_doc("Test article " * 30)
        doc["published_date"] = "March 15, 2024"
        result = self.cleaner.clean(doc)
        assert result is not None
        assert result["published_date"].startswith("2024-03-15")

    def test_clean_handles_already_iso_date(self):
        doc = self._make_doc("Test article " * 30)
        doc["published_date"] = "2024-03-15T10:00:00Z"
        result = self.cleaner.clean(doc)
        assert result is not None
        assert result["published_date"] == "2024-03-15T10:00:00Z"

    def test_clean_batch_filters_invalid(self):
        docs = [
            self._make_doc("Short.", "doc_001"),
            self._make_doc("Valid article about AI funding rounds and LLM startups. " * 10, "doc_002"),
            self._make_doc("", "doc_003"),
        ]
        results = self.cleaner.clean_batch(docs)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc_002"

    def test_clean_does_not_mutate_original(self):
        original_text = "AI startup raises funding. " * 20
        doc = self._make_doc(original_text)
        self.cleaner.clean(doc)
        assert doc["raw_text"] == original_text
        assert doc["cleaned_text"] == ""


class TestDeduplicator:

    def setup_method(self):
        self.dedup = Deduplicator(threshold=0.85)

    def _make_doc(self, text: str, doc_id: str) -> dict:
        return {
            "doc_id": doc_id,
            "source_url": f"https://example.com/{doc_id}",
            "publisher": "Test",
            "raw_text": text,
            "cleaned_text": text,
        }

    def test_keeps_unique_documents(self):
        docs = [
            self._make_doc("Anthropic raises $2 billion in AI funding round led by Google.", "doc_1"),
            self._make_doc("OpenAI announces GPT-5 with improved reasoning capabilities.", "doc_2"),
            self._make_doc("Mistral AI launches new open-source language model for enterprise.", "doc_3"),
        ]
        result = self.dedup.deduplicate(docs)
        assert len(result) == 3

    def test_removes_exact_duplicate(self):
        text = (
            "Anthropic has raised $2 billion in new funding led by Google and Spark Capital. "
            "The company plans to expand its AI research and scale Claude deployments." * 3
        )
        docs = [
            self._make_doc(text, "doc_1"),
            self._make_doc(text, "doc_2"),
        ]
        result = self.dedup.deduplicate(docs)
        assert len(result) == 1
        assert result[0]["doc_id"] == "doc_1"

    def test_preserves_order_of_first_occurrence(self):
        text = "Same article text repeated. " * 20
        docs = [
            self._make_doc(text, "first"),
            self._make_doc(text, "second"),
            self._make_doc(text, "third"),
        ]
        result = self.dedup.deduplicate(docs)
        assert result[0]["doc_id"] == "first"

    def test_skips_empty_text_docs(self):
        docs = [{"doc_id": "empty", "source_url": "https://x.com", "publisher": "T", "raw_text": "", "cleaned_text": ""}]
        result = self.dedup.deduplicate(docs)
        assert len(result) == 0

    def test_reset_clears_index(self):
        text = "Long enough article to be indexed. " * 20
        doc = self._make_doc(text, "doc_1")
        self.dedup.deduplicate([doc])
        self.dedup.reset()
        result = self.dedup.deduplicate([doc])
        assert len(result) == 1


class TestTokenize:

    def test_produces_3_shingles(self):
        text = "the quick brown fox jumps"
        shingles = _tokenize(text)
        assert len(shingles) > 0
        for shingle in shingles:
            words = shingle.decode("utf-8").split()
            assert len(words) == 3

    def test_handles_short_text(self):
        shingles = _tokenize("hi")
        assert isinstance(shingles, list)

    def test_is_lowercase(self):
        shingles = _tokenize("Anthropic Raises Funding")
        for shingle in shingles:
            assert shingle.decode("utf-8") == shingle.decode("utf-8").lower()
