"""
src/scraper/hackernews_scraper.py

Hacker News scraper using the official public Algolia API.
No authentication required, no robots.txt issues, completely free.

Targets "Show HN" and top stories mentioning AI startup keywords.
The Algolia API returns structured JSON — no HTML parsing needed.

API docs: https://hn.algolia.com/api
"""

import logging
import time
from datetime import datetime, timezone

import requests

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# Algolia HN Search API — public, free, no auth
HN_API_BASE = "https://hn.algolia.com/api/v1"

# Search queries to run against HN
HN_QUERIES = [
    "AI startup funding",
    "LLM startup",
    "generative AI company",
    "machine learning funding round",
    "AI company series",
    "foundation model startup",
    "vector database",
    "RAG retrieval augmented",
    "AI engineer hiring",
    "artificial intelligence investment",
]

AI_KEYWORDS = {
    "ai", "artificial intelligence", "machine learning", "llm", "gpt",
    "startup", "funding", "series", "generative", "foundation model",
    "openai", "anthropic", "mistral", "vector database", "rag",
    "neural", "transformer", "deep learning", "nlp", "diffusion",
}


class HackerNewsScraper(BaseScraper):
    """
    Scrapes AI startup stories from Hacker News via the Algolia API.
    Returns structured story data without needing to parse HTML.
    """

    def __init__(self):
        super().__init__(
            source_name="HackerNews",
            base_url="https://news.ycombinator.com"
        )
        self._api_session = requests.Session()
        self._api_session.headers.update({
            "User-Agent": "TrendScoutBot/1.0 (Academic Research; ASU CSE573)"
        })

    def fetch_article_list(self, max_articles: int) -> list:
        """
        Returns HN story IDs (as URLs) for AI-relevant stories.
        We store story data during this phase to avoid double-fetching.
        """
        self._story_cache = {}
        urls = []
        seen_ids = set()
        per_query = max(max_articles // len(HN_QUERIES), 5)

        for query in HN_QUERIES:
            if len(urls) >= max_articles:
                break
            try:
                stories = self._search_stories(query, per_query)
                for story in stories:
                    story_id = str(story.get("objectID", ""))
                    if not story_id or story_id in seen_ids:
                        continue
                    if not self._is_ai_relevant(story):
                        continue
                    # Use HN story URL as the document URL
                    url = f"https://news.ycombinator.com/item?id={story_id}"
                    urls.append(url)
                    seen_ids.add(story_id)
                    self._story_cache[url] = story
                time.sleep(0.5)  # Polite delay between API calls
            except Exception as e:
                logger.warning(f"[HackerNews] Query '{query}' failed: {e}")

        logger.info(f"[HackerNews] Found {len(urls)} relevant stories")
        return urls[:max_articles]

    def parse_article(self, html: str, url: str) -> dict:
        """
        For HN we use cached API data instead of parsing HTML.
        The API gives us clean structured data.
        """
        story = self._story_cache.get(url)
        if not story:
            return {}

        title = story.get("title", "")
        text = story.get("story_text") or story.get("text") or ""
        author = story.get("author", "")
        points = story.get("points", 0)
        num_comments = story.get("num_comments", 0)
        created_at = story.get("created_at", "")
        external_url = story.get("url", "")

        # Build rich text combining title, story text, and context
        raw_text = (
            f"{title}\n\n"
            f"{text}\n\n"
            f"Points: {points} | Comments: {num_comments}\n"
            f"External URL: {external_url}"
        ).strip()

        if not title:
            return {}

        # Minimum content check
        if len(raw_text) < 100:
            raw_text = title * 3  # Repeat title to pass length filter

        return {
            "title": title,
            "raw_text": raw_text,
            "published_date": created_at,
            "author": author or None,
            "tags": self._extract_tags(title + " " + text),
            "metadata": {
                "points": points,
                "num_comments": num_comments,
                "external_url": external_url,
                "document_type": "hn_story",
            },
        }

    # ── Private helpers ──────────────────────────────────────

    def _search_stories(self, query: str, num_results: int) -> list:
        """Query the Algolia HN Search API."""
        params = {
            "query": query,
            "tags": "story",
            "hitsPerPage": min(num_results * 2, 50),  # Get extra, filter later
            "attributesToRetrieve": (
                "objectID,title,story_text,author,points,"
                "num_comments,created_at,url"
            ),
        }
        response = self._api_session.get(
            f"{HN_API_BASE}/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("hits", [])

    def _is_ai_relevant(self, story: dict) -> bool:
        """Filter to AI/startup-relevant stories."""
        text = (
            story.get("title", "") + " " +
            (story.get("story_text") or "")
        ).lower()
        return any(kw in text for kw in AI_KEYWORDS)

    def _extract_tags(self, text: str) -> list:
        """Extract relevant topic tags from story text."""
        tag_keywords = [
            "LLM", "RAG", "RLHF", "GPT", "AI", "ML",
            "startup", "funding", "Series A", "Series B",
            "open source", "transformer", "vector database",
            "NLP", "computer vision", "robotics",
        ]
        text_lower = text.lower()
        return [
            tag for tag in tag_keywords
            if tag.lower() in text_lower
        ][:10]
