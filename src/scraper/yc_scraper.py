import logging
import re
from urllib.parse import urljoin
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

YC_COMPANIES_URL = "https://www.ycombinator.com/companies"


class YCScraper(BaseScraper):

    def __init__(self):
        super().__init__(source_name="Y Combinator", base_url="https://www.ycombinator.com")

    def fetch_article_list(self, max_articles: int) -> list:
        urls = []
        html = self._get_with_retry(YC_COMPANIES_URL)
        if not html:
            logger.warning("[YC] Could not reach YC directory")
            return []
        soup = self._make_soup(html)
        for a in soup.find_all("a", href=re.compile(r"^/companies/[a-z0-9\-]+")):
            href = a["href"]
            full_url = urljoin(self.base_url, href)
            if full_url not in urls:
                urls.append(full_url)
            if len(urls) >= max_articles:
                break
        logger.info(f"[YC] Found {len(urls)} company URLs")
        return urls

    def parse_article(self, html: str, url: str) -> dict:
        soup = self._make_soup(html)
        name = self._extract_text(soup, [
            "h1[class*='company-name']",
            "h1[class*='CompanyName']",
            "h1",
        ])
        tagline = self._extract_text(soup, [
            "p[class*='tagline']",
            "p[class*='one-liner']",
            "div[class*='tagline']",
        ])
        description = self._extract_text(soup, [
            "div[class*='long-description']",
            "div[class*='description']",
            "section[class*='about']",
        ])
        batch = self._extract_text(soup, [
            "span[class*='batch']",
            "a[class*='batch']",
        ])
        tags = [
            span.get_text(strip=True)
            for span in soup.select("span[class*='tag'], a[class*='tag']")
        ]
        website_el = soup.find("a", string=re.compile(r"website|visit", re.I))
        website = website_el["href"] if website_el else ""
        if not name:
            return {}
        raw_text = f"{name}\n{tagline}\n\n{description}".strip()
        if not raw_text:
            return {}
        return {
            "title": name,
            "raw_text": raw_text,
            "published_date": None,
            "author": None,
            "tags": tags[:10],
            "metadata": {
                "batch": batch,
                "website": website,
                "tagline": tagline,
            },
        }
