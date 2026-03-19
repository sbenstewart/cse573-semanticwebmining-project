import re
import logging
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LDA_NUM_TOPICS, LDA_PASSES, LDA_RANDOM_STATE, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = PROCESSED_DATA_DIR / "lda_model"
DICT_PATH  = PROCESSED_DATA_DIR / "lda_dictionary.pkl"

DOMAIN_STOPWORDS = {
    "said", "says", "also", "would", "could", "one", "two", "three",
    "new", "use", "used", "using", "year", "years", "company", "companies",
    "startup", "startups", "technology", "technologies", "product", "products",
    "team", "people", "work", "make", "made", "get", "like", "well",
    "according", "announced", "today", "week", "month",
}

ALL_STOPWORDS = STOPWORDS | DOMAIN_STOPWORDS


def _preprocess(text: str) -> list:
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    return [t for t in tokens if t not in ALL_STOPWORDS]


class TopicModel:

    def __init__(self, num_topics: int = LDA_NUM_TOPICS):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None
        self.docs = []
        self.corpus_bow = []
        self._is_fitted = False

    def fit(self, docs: list) -> None:
        self.docs = docs
        texts = [
            _preprocess(d.get("cleaned_text") or d.get("raw_text", ""))
            for d in docs
        ]
        texts = [t for t in texts if len(t) >= 5]
        logger.info(f"[LDA] Building dictionary from {len(texts)} documents...")
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus_bow = [self.dictionary.doc2bow(t) for t in texts]
        logger.info(f"[LDA] Training LDA: {self.num_topics} topics, {LDA_PASSES} passes...")
        self.lda_model = models.LdaModel(
            corpus=self.corpus_bow,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=LDA_PASSES,
            random_state=LDA_RANDOM_STATE,
            alpha="auto",
            eta="auto",
            per_word_topics=True,
        )
        self._is_fitted = True
        logger.info("[LDA] Training complete.")

    def get_topics(self, top_n: int = 10) -> list:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        topics = []
        for topic_id in range(self.num_topics):
            word_probs = self.lda_model.show_topic(topic_id, topn=top_n)
            topics.append({
                "topic_id": topic_id,
                "top_words": [w for w, _ in word_probs],
                "top_word_probs": {w: float(p) for w, p in word_probs},
            })
        return topics

    def get_document_topics(self, doc: dict, min_prob: float = 0.05) -> list:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        text = _preprocess(doc.get("cleaned_text") or doc.get("raw_text", ""))
        bow = self.dictionary.doc2bow(text)
        topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=min_prob)
        return [
            {"topic_id": int(tid), "probability": float(prob)}
            for tid, prob in sorted(topic_dist, key=lambda x: -x[1])
        ]

    def print_topics(self, top_n: int = 10) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        print(f"\n{'='*60}")
        print(f"  TrendScout LDA - {self.num_topics} Topics")
        print(f"{'='*60}")
        for topic in self.get_topics(top_n):
            words = ", ".join(topic["top_words"])
            print(f"\n  Topic {topic['topic_id']:>2}: {words}")
        print(f"\n{'='*60}\n")

    def skill_trend_over_time(self, skills: list) -> dict:
        trends = {skill: defaultdict(int) for skill in skills}
        skills_lower = {s.lower(): s for s in skills}
        for doc in self.docs:
            text = (doc.get("cleaned_text") or doc.get("raw_text", "")).lower()
            date_str = doc.get("published_date", "")
            month = date_str[:7] if date_str else "unknown"
            for skill_lower, skill_orig in skills_lower.items():
                count = len(re.findall(r"\b" + re.escape(skill_lower) + r"\b", text))
                if count > 0:
                    trends[skill_orig][month] += count
        return {skill: dict(sorted(mc.items())) for skill, mc in trends.items()}

    def top_terms_frequency(self, top_n: int = 30) -> list:
        counter = Counter()
        for doc in self.docs:
            text = _preprocess(doc.get("cleaned_text") or doc.get("raw_text", ""))
            counter.update(text)
        return counter.most_common(top_n)

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.lda_model.save(str(MODEL_PATH))
        with open(DICT_PATH, "wb") as f:
            pickle.dump({"dictionary": self.dictionary, "docs": self.docs,
                         "corpus_bow": self.corpus_bow, "num_topics": self.num_topics}, f)
        logger.info(f"[LDA] Model saved to {MODEL_PATH}")

    def load(self) -> None:
        self.lda_model = models.LdaModel.load(str(MODEL_PATH))
        with open(DICT_PATH, "rb") as f:
            data = pickle.load(f)
        self.dictionary = data["dictionary"]
        self.docs = data["docs"]
        self.corpus_bow = data["corpus_bow"]
        self.num_topics = data["num_topics"]
        self._is_fitted = True
        logger.info(f"[LDA] Model loaded from {MODEL_PATH}")
