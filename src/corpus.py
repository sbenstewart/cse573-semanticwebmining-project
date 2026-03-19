import json
import logging
from pathlib import Path
import jsonlines

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CORPUS_FILE, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"doc_id", "source_url", "publisher", "raw_text"}

DEFAULTS = {
    "author": None,
    "published_date": None,
    "crawl_time": None,
    "title": "",
    "cleaned_text": "",
    "tags": [],
    "extraction_method": "unknown",
    "confidence": 1.0,
    "dominant_topic": None,
    "topic_probability": None,
    "topic_distribution": [],
    "metadata": {},
}


def validate_doc(doc: dict) -> bool:
    missing = REQUIRED_FIELDS - set(doc.keys())
    if missing:
        logger.warning(f"Document missing fields {missing}: {doc.get('doc_id', '?')}")
        return False
    return True


def normalize_doc(doc: dict) -> dict:
    result = dict(DEFAULTS)
    result.update(doc)
    return result


def save_docs(docs: list, path: Path = CORPUS_FILE) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with jsonlines.open(path, mode="a") as writer:
        for doc in docs:
            if not validate_doc(doc):
                continue
            doc = normalize_doc(doc)
            writer.write(doc)
            written += 1
    logger.info(f"[Corpus] Wrote {written} documents to {path}")
    return written


def load_docs(path: Path = CORPUS_FILE) -> list:
    if not path.exists():
        logger.warning(f"[Corpus] File not found: {path}")
        return []
    with jsonlines.open(path, mode="r") as reader:
        docs = list(reader)
    logger.info(f"[Corpus] Loaded {len(docs)} documents from {path}")
    return docs


def save_raw(docs: list, source_name: str) -> Path:
    path = RAW_DATA_DIR / f"{source_name.lower().replace(' ', '_')}.jsonl"
    with jsonlines.open(path, mode="w") as writer:
        for doc in docs:
            writer.write(doc)
    logger.info(f"[Corpus] Saved {len(docs)} raw docs to {path}")
    return path


def corpus_stats(docs: list) -> dict:
    publishers = {}
    date_range = []
    total_chars = 0
    has_cleaned = 0

    for doc in docs:
        pub = doc.get("publisher", "unknown")
        publishers[pub] = publishers.get(pub, 0) + 1
        dt = doc.get("published_date")
        if dt:
            date_range.append(dt[:10])
        total_chars += len(doc.get("cleaned_text") or doc.get("raw_text", ""))
        if doc.get("cleaned_text"):
            has_cleaned += 1

    date_range.sort()
    return {
        "total_documents": len(docs),
        "by_publisher": publishers,
        "avg_text_length": int(total_chars / max(len(docs), 1)),
        "documents_with_cleaned_text": has_cleaned,
        "date_range": {
            "earliest": date_range[0] if date_range else None,
            "latest": date_range[-1] if date_range else None,
        },
    }
