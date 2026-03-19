import logging
import feedparser
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

AI_KEYWORDS = {
    "ai", "artificial intelligence", "machine learning", "llm", "gpt",
    "startup", "funding", "seed", "series a", "series b", "venture",
    "generative", "foundation model", "openai", "anthropic", "mistral",
    "vector database", "rag", "neural", "transformer", "vc", "investment",
}


class TechCrunchScraper(BaseScraper):

    SECTION_FEEDS = [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://techcrunch.com/category/startups/feed/",
        "https://techcrunch.com/feed/",
    ]

    def __init__(self):
        super().__init__(source_name="TechCrunch", base_url="https://techcrunch.com")

    def fetch_article_list(self, max_articles: int) -> list:
        urls = []
        seen = set()
        for feed_url in self.SECTION_FEEDS:
            if len(urls) >= max_articles:
                break
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    url = entry.get("link", "")
                    if not url or url in seen:
                        continue
                    if self._is_ai_relevant(entry):
                        urls.append(url)
                        seen.add(url)
            except Exception as e:
                logger.warning(f"[TechCrunch] RSS error {feed_url}: {e}")
        return urls[:max_articles]

    def parse_article(self, html: str, url: str) -> dict:
        soup = self._make_soup(html)
        title = self._extract_text(soup, [
            "h1.wp-block-post-title",
            "h1[class*='article-hero-title']",
            "h1",
        ])
        body_el = (
            soup.select_one("div.wp-block-post-content") or
            soup.select_one("div[class*='article-content']") or
            soup.select_one("article")
        )
        raw_text = body_el.get_text(separator=" ", strip=True) if body_el else ""
        time_el = soup.find("time", attrs={"datetime": True})
        published_date = time_el["datetime"] if time_el else None
        author = self._extract_text(soup, [
            "a[class*='author-name']",
            "span[class*='author']",
            "a[rel='author']",
        ])
        tags = [a.get_text(strip=True) for a in soup.select("a[class*='tag'], a[class*='category']")]
        if not raw_text or not title:
            return {}
        return {
            "title": title,
            "raw_text": f"{title}\n\n{raw_text}",
            "published_date": published_date,
            "author": author or None,
            "tags": tags[:10],
        }

    def _is_ai_relevant(self, entry: dict) -> bool:
        text = (
            entry.get("title", "") + " " +
            entry.get("summary", "") + " " +
            " ".join(tag.get("term", "") for tag in entry.get("tags", []))
        ).lower()
        return any(kw in text for kw in AI_KEYWORDS)
