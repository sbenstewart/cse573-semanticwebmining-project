import uuid
from datetime import datetime

class StartupDocument:
    """Standardized schema for all ingested documents[cite: 75]."""
    def __init__(self, raw_text, source_url, publisher, author=None, published_date=None):
        self.doc_id = str(uuid.uuid4())
        self.source_url = source_url
        self.publisher = publisher
        self.author = author
        self.published_date = published_date or datetime.now().isoformat()
        self.crawl_time = datetime.now().isoformat()
        self.raw_text = raw_text
        self.cleaned_text = self._clean_text(raw_text)

    def _clean_text(self, text):
        """Removes boilerplate and normalizes text[cite: 70]."""
        import re
        # Basic cleaning: remove extra whitespace and potential HTML tags
        clean = re.sub(r'\s+', ' ', text)
        clean = re.sub(r'<.*?>', '', clean)
        return clean.strip()

    def to_dict(self):
        return self.__dict__