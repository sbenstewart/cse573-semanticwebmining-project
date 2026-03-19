import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CORPUS_FILE = ROOT_DIR / os.getenv("CORPUS_FILE", "data/processed/corpus.jsonl")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

SCRAPER_RATE_LIMIT = float(os.getenv("SCRAPER_RATE_LIMIT_SECONDS", 2.0))
SCRAPER_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", 3))
SCRAPER_USER_AGENT = os.getenv("SCRAPER_USER_AGENT", "TrendScoutBot/1.0 (Academic Research; ASU CSE573)")
SCRAPER_HEADERS = {
    "User-Agent": SCRAPER_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

SOURCES = {
    "techcrunch":  {"name": "TechCrunch",  "rss_url": "https://techcrunch.com/feed/",      "category": "tech_news"},
    "venturebeat": {"name": "VentureBeat", "rss_url": "https://venturebeat.com/feed/",     "category": "tech_news"},
    "yc":          {"name": "Y Combinator","base_url": "https://www.ycombinator.com",      "category": "startup_listings"},
}

TFIDF_MAX_FEATURES = 10_000
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K_RESULTS = 5

LDA_NUM_TOPICS = 12
LDA_PASSES = 20
LDA_RANDOM_STATE = 42

MINHASH_NUM_PERM = 128
MINHASH_THRESHOLD = 0.85

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
