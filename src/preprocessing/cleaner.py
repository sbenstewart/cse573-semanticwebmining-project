import re
import logging
import unicodedata
from datetime import datetime

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 200

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
        text = _BOILERPLATE_RE.sub(" ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"[^\w\s\.,;:!?\-\(\)\[\]\"\'\/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not self._is_english(text):
            return None
        if len(text) < MIN_TEXT_LENGTH:
            return None

        doc = dict(doc)
        doc["cleaned_text"] = text
        doc["published_date"] = self._normalize_date(doc.get("published_date"))
        return doc

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
