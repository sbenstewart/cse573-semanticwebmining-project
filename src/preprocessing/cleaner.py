import re
import logging
import unicodedata
from datetime import datetime
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 200

# Matches "Headline: X. Summary: ..." or "Job Title: X. Description: ..."
_TITLE_RE = re.compile(
    r"^\s*(?:Headline|Job\s*Title|Title)\s*:\s*(.+?)\s*(?:\.\s+(?:Summary|Description|Body|Abstract)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)

BOILERPLATE_PATTERNS = [
    r"Subscribe to .{0,60} newsletter",
    r"Sign up for .{0,60} newsletter",
    r"Copyright © \d{4}",
    r"All rights reserved",
    r"Terms of (Service|Use)",
    r"Privacy Policy",
    r"Cookie (Policy|Settings)",
    r"Advertisement",
    r"Follow us on (Twitter|LinkedIn|Facebook|Instagram)",
    r"Read more:",
    r"Also read:",
    r"Related articles?:",
    r"You might also like:",
]

_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


class TextCleaner:

    def clean(self, doc: dict):
        raw = doc.get("raw_text", "")
        if not raw:
            return None

        text = unicodedata.normalize("NFKC", raw)
        # Strip HTML tags first (handles full and partial tags like "<br>", "br /", "/li")
        text = BeautifulSoup(text, "lxml").get_text(separator=" ")
        text = re.sub(r"</?\s*[a-zA-Z][a-zA-Z0-9]*\s*/?>", " ", text)
        text = _BOILERPLATE_RE.sub(" ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"[^\w\s\.,;:!?\-\(\)\[\]\"\'\/]", " ", text)
        # Remove dangling tag-name tokens left over from malformed markup
        text = re.sub(r"\b(br|li|ul|ol|strong|em|div|span|p|h[1-6])\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()

        if not self._is_english(text):
            return None
        if len(text) < MIN_TEXT_LENGTH:
            return None

        # Strip the "Headline: X. Summary:" / "Job Title: X. Description:" artifact prefix
        # AFTER the length check (some feeds are short and would otherwise get dropped),
        # so it doesn't pollute BM25/TF-IDF scoring with boilerplate labels.
        text = re.sub(
            r"^\s*(?:Headline|Job\s*Title|Title)\s*:\s*.+?\.\s+(?:Summary|Description|Body|Abstract)\s*:\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

        doc = dict(doc)
        doc["cleaned_text"] = text
        # Extract a title if the scraper didn't provide one
        if not doc.get("title"):
            doc["title"] = self._extract_title(raw, text)
        doc["published_date"] = self._normalize_date(doc.get("published_date"))
        return doc

    @staticmethod
    def _extract_title(raw: str, cleaned: str) -> str:
        # Try "Headline: ... Summary:" / "Job Title: ... Description:" prefixes
        m = _TITLE_RE.match(raw)
        if m:
            title = m.group(1).strip()
            # Truncate trailing source attribution like " - Publisher Name"
            title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()
            if title:
                return title[:200]
        # Fallback: first sentence of cleaned text, capped
        first = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
        return first[:120].strip()

    def clean_batch(self, docs: list) -> list:
        cleaned = []
        for doc in docs:
            result = self.clean(doc)
            if result:
                cleaned.append(result)
        n_dropped = len(docs) - len(cleaned)
        logger.info(f"[Cleaner] {len(cleaned)}/{len(docs)} docs kept ({n_dropped} dropped)")
        return cleaned

    @staticmethod
    def _is_english(text: str) -> bool:
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return False
        ascii_ratio = sum(1 for c in alpha_chars if ord(c) < 128) / len(alpha_chars)
        return ascii_ratio > 0.80

    @staticmethod
    def _normalize_date(date_str):
        if not date_str:
            return None
        if re.match(r"\d{4}-\d{2}-\d{2}T", date_str):
            return date_str
        # Try RFC-2822 first (e.g. "Wed, 29 Oct 2025 12:34:56 GMT")
        try:
            dt = parsedate_to_datetime(date_str)
            if dt is not None:
                return dt.isoformat()
        except (TypeError, ValueError):
            pass
        formats = [
            "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y",
            "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip()[:20], fmt)
                return dt.isoformat()
            except ValueError:
                continue
        return date_str