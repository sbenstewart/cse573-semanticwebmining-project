"""
src/scraper/arxiv_scraper.py

arXiv scraper using the official public API.
No authentication required, explicitly allows programmatic access.

Targets recent AI/ML papers relevant to startup technology trends:
  - Large language models
  - Retrieval augmented generation
  - Knowledge graphs
  - AI agents and systems

API docs: https://arxiv.org/help/api/user-manual
"""

import logging
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# Search queries targeting startup-relevant AI research
ARXIV_QUERIES = [
    "large language model startup deployment",
    "retrieval augmented generation enterprise",
    "knowledge graph question answering",
    "LLM fine-tuning production",
    "vector database similarity search",
    "AI agent autonomous systems",
    "foundation model evaluation benchmark",
    "transformer NER information extraction",
]

# arXiv categories to focus on
AI_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.IR"]


class ArXivScraper(BaseScraper):
    """
    Scrapes recent AI/ML papers from arXiv via the official Atom/XML API.
    Provides high-quality technical content about AI technologies
    used by startups.
    """

    def __init__(self):
        super().__init__(
            source_name="arXiv",
            base_url="https://arxiv.org"
        )
        self._paper_cache = {}

    def fetch_article_list(self, max_articles: int) -> list:
        """Fetch paper URLs from arXiv API."""
        urls = []
        seen = set()
        per_query = max(max_articles // len(ARXIV_QUERIES), 3)

        for query in ARXIV_QUERIES:
            if len(urls) >= max_articles:
                break
            try:
                papers = self._search_papers(query, per_query)
                for paper in papers:
                    url = paper.get("url", "")
                    if url and url not in seen:
                        urls.append(url)
                        seen.add(url)
                        self._paper_cache[url] = paper
                time.sleep(1.0)  # arXiv requests 1s delay between calls
            except Exception as e:
                logger.warning(f"[arXiv] Query '{query}' failed: {e}")

        logger.info(f"[arXiv] Found {len(urls)} papers")
        return urls[:max_articles]

    def parse_article(self, html: str, url: str) -> dict:
        """Use cached paper data from API response."""
        paper = self._paper_cache.get(url)
        if not paper:
            return {}

        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        authors = paper.get("authors", [])
        published = paper.get("published", "")
        categories = paper.get("categories", [])

        if not title or not abstract:
            return {}

        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += f" et al."

        raw_text = (
            f"{title}\n\n"
            f"Authors: {author_str}\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Abstract: {abstract}"
        )

        return {
            "title": title,
            "raw_text": raw_text,
            "published_date": published,
            "author": author_str or None,
            "tags": categories[:5],
            "metadata": {
                "authors": authors,
                "categories": categories,
                "document_type": "research_paper",
                "arxiv_url": url,
            },
        }

    # ── Private helpers ──────────────────────────────────────

    def _search_papers(self, query: str, max_results: int) -> list:
        """Query the arXiv API and parse Atom XML response."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_BASE}?{urlencode(params)}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        papers = []

        for entry in soup.find_all("entry"):
            try:
                paper_id = entry.find("id").text.strip()
                title = entry.find("title").text.strip().replace("\n", " ")
                abstract = entry.find("summary").text.strip().replace("\n", " ")
                published = entry.find("published").text.strip()
                authors = [
                    a.find("name").text.strip()
                    for a in entry.find_all("author")
                    if a.find("name")
                ]
                categories = [
                    c.get("term", "")
                    for c in entry.find_all("category")
                ]

                # Filter to AI categories
                if not any(cat in AI_CATEGORIES for cat in categories):
                    continue

                papers.append({
                    "url": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "published": published,
                    "authors": authors,
                    "categories": categories,
                })
            except Exception as e:
                logger.debug(f"[arXiv] Failed to parse entry: {e}")
                continue

        return papers
