"""
src/scraper/base_scraper.py

Abstract base scraper with:
  - Polite rate limiting
  - Retry with exponential backoff
  - robots.txt compliance check
  - Provenance metadata capture
  - Structured document output (matches corpus schema)
"""

import time
import uuid
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Import settings — adjust path if running from project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SCRAPER_HEADERS, SCRAPER_RATE_LIMIT, SCRAPER_MAX_RETRIES

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Abstract base class for all TrendScout scrapers.

    Subclasses must implement:
        - fetch_article_list() -> list[str]   : returns URLs to scrape
        - parse_article(html, url) -> dict    : extracts fields from raw HTML
    """

    def __init__(self, source_name: str, base_url: str):
        self.source_name = source_name
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(SCRAPER_HEADERS)
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._last_request_time: float = 0.0

    # ── Public Interface ─────────────────────────────────────

    def scrape(self, max_articles: int = 50) -> list[dict]:
        """
        Main entry point. Returns a list of document dicts ready
        to be written to the corpus.
        """
        logger.info(f"[{self.source_name}] Starting scrape (max={max_articles})")
        urls = self.fetch_article_list(max_articles)
        logger.info(f"[{self.source_name}] Found {len(urls)} URLs")

        documents = []
        for url in urls[:max_articles]:
            if not self._is_allowed(url):
                logger.warning(f"[{self.source_name}] robots.txt blocks: {url}")
                continue

            html = self._get_with_retry(url)
            if html is None:
                continue

            try:
                doc = self.parse_article(html, url)
                if doc:
                    doc = self._enrich_provenance(doc, url)
                    documents.append(doc)
                    logger.debug(f"[{self.source_name}] Parsed: {url}")
            except Exception as e:
                logger.error(f"[{self.source_name}] Parse error for {url}: {e}")

        logger.info(f"[{self.source_name}] Scraped {len(documents)} documents")
        return documents

    # ── Abstract Methods ─────────────────────────────────────

    @abstractmethod
    def fetch_article_list(self, max_articles: int) -> list[str]:
        """Return a list of article URLs to scrape."""
        ...

    @abstractmethod
    def parse_article(self, html: str, url: str) -> dict:
        """
        Parse raw HTML and return a document dict with at minimum:
            title, raw_text, published_date, author (optional), tags (optional)
        """
        ...

    # ── Internal Helpers ─────────────────────────────────────

    def _get_with_retry(self, url: str) -> str | None:
        """HTTP GET with rate limiting and exponential backoff retry."""
        for attempt in range(1, SCRAPER_MAX_RETRIES + 1):
            self._rate_limit()
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except requests.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                if status == 429:
                    wait = 2 ** attempt * 5
                    logger.warning(f"Rate limited on {url}. Waiting {wait}s...")
                    time.sleep(wait)
                elif status in (403, 404):
                    logger.warning(f"HTTP {status} — skipping {url}")
                    return None
                else:
                    logger.warning(f"HTTP {status} on attempt {attempt} for {url}")
            except requests.RequestException as e:
                logger.warning(f"Request error attempt {attempt} for {url}: {e}")
                time.sleep(2 ** attempt)

        logger.error(f"Failed after {SCRAPER_MAX_RETRIES} attempts: {url}")
        return None

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SCRAPER_RATE_LIMIT:
            time.sleep(SCRAPER_RATE_LIMIT - elapsed)
        self._last_request_time = time.time()

    def _is_allowed(self, url: str) -> bool:
        """Check robots.txt for the given URL."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if robots_url not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, assume allowed
                rp = None
            self._robots_cache[robots_url] = rp

        rp = self._robots_cache[robots_url]
        if rp is None:
            return True
        return rp.can_fetch(SCRAPER_HEADERS["User-Agent"], url)

    def _enrich_provenance(self, doc: dict, url: str) -> dict:
        """
        Add provenance fields required by the corpus schema.
        Fills in defaults for any missing fields.
        """
        crawl_time = datetime.now(timezone.utc).isoformat()

        # Deterministic doc_id from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        source_prefix = self.source_name[:3].lower()
        pub_date = doc.get("published_date", "")[:10].replace("-", "") or "00000000"
        doc_id = f"{source_prefix}_{pub_date}_{url_hash}"

        return {
            "doc_id": doc_id,
            "source_url": url,
            "publisher": self.source_name,
            "author": doc.get("author", None),
            "published_date": doc.get("published_date", None),
            "crawl_time": crawl_time,
            "title": doc.get("title", ""),
            "raw_text": doc.get("raw_text", ""),
            "cleaned_text": "",        # Filled by preprocessing pipeline
            "tags": doc.get("tags", []),
            "extraction_method": "rule-based-scraper",
            "confidence": 1.0,        # Raw scrape = full confidence on provenance
        }

    # ── Utility ──────────────────────────────────────────────

    @staticmethod
    def _extract_text(soup: BeautifulSoup, selectors: list[str]) -> str:
        """
        Try a list of CSS selectors in order, return the first match's text.
        Used to handle sites that change their HTML structure.
        """
        for selector in selectors:
            el = soup.select_one(selector)
            if el:
                return el.get_text(separator=" ", strip=True)
        return ""

    @staticmethod
    def _make_soup(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "lxml")
