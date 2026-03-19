import re
import logging
from datasketch import MinHash, MinHashLSH

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MINHASH_NUM_PERM, MINHASH_THRESHOLD

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < 3:
        return [t.encode("utf-8") for t in tokens]
    shingles = [
        " ".join(tokens[i:i+3]).encode("utf-8")
        for i in range(len(tokens) - 2)
    ]
    return shingles


def _make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _tokenize(text):
        m.update(shingle)
    return m


class Deduplicator:

    def __init__(self, threshold: float = MINHASH_THRESHOLD, num_perm: int = MINHASH_NUM_PERM):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._doc_hashes = {}

    def deduplicate(self, docs: list) -> list:
        unique = []
        n_dupes = 0

        for doc in docs:
            text = doc.get("cleaned_text") or doc.get("raw_text", "")
            if not text:
                continue

            doc_id = doc["doc_id"]
            mh = _make_minhash(text, self.num_perm)

            try:
                results = self.lsh.query(mh)
            except Exception:
                results = []

            if results:
                n_dupes += 1
                logger.debug(f"[Dedup] Dropping duplicate {doc_id} (matches {results[0]})")
                continue

            try:
                self.lsh.insert(doc_id, mh)
                self._doc_hashes[doc_id] = mh
                unique.append(doc)
            except ValueError:
                logger.debug(f"[Dedup] Skipping already-indexed: {doc_id}")

        logger.info(f"[Dedup] {len(unique)}/{len(docs)} unique docs kept ({n_dupes} duplicates removed)")
        return unique

    def reset(self):
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._doc_hashes.clear()
