import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper.base_scraper import BaseScraper
from src.scraper.techcrunch import TechCrunchScraper
from src.scraper.job_scraper import JobScraper


class MockScraper(BaseScraper):
    def __init__(self):
        super().__init__(source_name="TestSource", base_url="https://example.com")

    def fetch_article_list(self, max_articles: int) -> list:
        return ["https://example.com/article/1", "https://example.com/article/2"]

    def parse_article(self, html: str, url: str) -> dict:
        return {
            "title": "Test Article",
            "raw_text": "This is a test article about AI startups and LLM funding.",
            "published_date": "2024-03-15T10:00:00Z",
            "author": "Test Author",
            "tags": ["AI", "startup"],
        }


class TestBaseScraper:

    def setup_method(self):
        self.scraper = MockScraper()

    def test_enrich_provenance_adds_required_fields(self):
        doc = {"title": "Test", "raw_text": "Hello world.", "published_date": "2024-03-15", "author": "Jane", "tags": []}
        enriched = self.scraper._enrich_provenance(doc, "https://example.com/test")
        assert "doc_id" in enriched
        assert "source_url" in enriched
        assert "crawl_time" in enriched
        assert "publisher" in enriched
        assert enriched["publisher"] == "TestSource"
        assert enriched["confidence"] == 1.0

    def test_doc_id_is_deterministic(self):
        doc = {"title": "T", "raw_text": "X", "published_date": "2024-01-01", "tags": []}
        url = "https://example.com/stable"
        id1 = self.scraper._enrich_provenance(doc, url)["doc_id"]
        id2 = self.scraper._enrich_provenance(doc, url)["doc_id"]
        assert id1 == id2

    def test_doc_id_differs_for_different_urls(self):
        doc = {"title": "T", "raw_text": "X", "published_date": "2024-01-01", "tags": []}
        id1 = self.scraper._enrich_provenance(doc, "https://example.com/a")["doc_id"]
        id2 = self.scraper._enrich_provenance(doc, "https://example.com/b")["doc_id"]
        assert id1 != id2

    def test_extract_text_returns_first_match(self):
        from bs4 import BeautifulSoup
        html = "<div><h1>Title Here</h1><p>Body</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = BaseScraper._extract_text(soup, ["h1", "p"])
        assert result == "Title Here"

    def test_extract_text_falls_back_to_second_selector(self):
        from bs4 import BeautifulSoup
        html = "<div><p>Body text</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = BaseScraper._extract_text(soup, ["h1", "p"])
        assert result == "Body text"

    def test_extract_text_returns_empty_if_no_match(self):
        from bs4 import BeautifulSoup
        html = "<div><span>Hello</span></div>"
        soup = BeautifulSoup(html, "lxml")
        result = BaseScraper._extract_text(soup, ["h1", "h2"])
        assert result == ""


class TestTechCrunchScraper:

    def setup_method(self):
        self.scraper = TechCrunchScraper()

    def test_is_ai_relevant_matches_ai_keyword(self):
        entry = {"title": "OpenAI raises $100M Series B", "summary": "", "tags": []}
        assert self.scraper._is_ai_relevant(entry) is True

    def test_is_ai_relevant_rejects_unrelated(self):
        entry = {"title": "Sports news: football match highlights", "summary": "", "tags": []}
        assert self.scraper._is_ai_relevant(entry) is False

    def test_is_ai_relevant_matches_llm_in_summary(self):
        entry = {"title": "New product launch", "summary": "Company integrates LLM into workflow.", "tags": []}
        assert self.scraper._is_ai_relevant(entry) is True

    def test_parse_article_returns_empty_for_missing_body(self):
        html = "<html><head></head><body><h1>Title</h1></body></html>"
        result = self.scraper.parse_article(html, "https://techcrunch.com/test")
        assert result == {}

    def test_parse_article_extracts_title_and_body(self):
        html = """
        <html><body>
          <h1>Anthropic raises $2B</h1>
          <div class="wp-block-post-content">
            Anthropic has raised $2 billion led by Google to expand LLM research.
          </div>
        </body></html>
        """
        result = self.scraper.parse_article(html, "https://techcrunch.com/test")
        assert result.get("title") == "Anthropic raises $2B"
        assert "Anthropic" in result.get("raw_text", "")


class TestJobScraper:

    def setup_method(self):
        self.scraper = JobScraper()

    def test_is_ai_job_matches_llm_engineer(self):
        assert self.scraper._is_ai_job("LLM Engineer", "") is True

    def test_is_ai_job_matches_in_description(self):
        assert self.scraper._is_ai_job("Software Engineer", "We are building machine learning infrastructure.") is True

    def test_is_ai_job_rejects_unrelated(self):
        assert self.scraper._is_ai_job("Office Manager", "Responsible for scheduling and office supplies.") is False

    def test_extract_skills_finds_python(self):
        desc = "Requirements: 5+ years Python, PyTorch, and CUDA experience."
        skills = self.scraper._extract_skills(desc)
        assert "Python" in skills
        assert "PyTorch" in skills
        assert "CUDA" in skills

    def test_extract_company_greenhouse(self):
        url = "https://boards.greenhouse.io/anthropic/jobs/12345"
        company = self.scraper._extract_company_from_url(url)
        assert company == "Anthropic"

    def test_extract_company_lever(self):
        url = "https://jobs.lever.co/scale-ai/abc-def"
        company = self.scraper._extract_company_from_url(url)
        assert "Scale" in company

    def test_extract_company_unknown(self):
        url = "https://mycompany.com/careers/job/123"
        company = self.scraper._extract_company_from_url(url)
        assert company == "Unknown"
