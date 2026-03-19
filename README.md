# TrendScout AI 2.0
**Knowledge Graph–Augmented Market Intelligence for AI Startups**

Arizona State University — CSE 573 Semantic Web Mining

---

## Project Overview

TrendScout AI 2.0 is a semantic web mining pipeline that monitors the AI startup ecosystem. It combines multi-source web mining, classical IR baselines, Knowledge Graph construction (Neo4j), and a KG-RAG conversational interface.

## Repository Structure

```
trendscout/
├── config/                  # Configuration files
│   └── settings.py
├── data/
│   ├── raw/                 # Raw scraped HTML/JSON
│   └── processed/           # Cleaned, deduplicated corpus JSON
├── src/
│   ├── scraper/             # Phase 1: Web scraping & data collection
│   │   ├── __init__.py
│   │   ├── techcrunch.py
│   │   ├── venturebeat.py
│   │   ├── yc_scraper.py
│   │   └── job_scraper.py
│   ├── preprocessing/       # Phase 1: Cleaning & deduplication
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   └── deduplicator.py
│   ├── retrieval/           # Phase 1: Classical IR baselines
│   │   ├── __init__.py
│   │   ├── tfidf_retriever.py
│   │   └── bm25_retriever.py
│   └── baseline/            # Phase 1: LDA topic modeling
│       ├── __init__.py
│       └── topic_model.py
├── tests/                   # Unit tests
│   ├── test_scraper.py
│   ├── test_preprocessing.py
│   └── test_retrieval.py
├── notebooks/               # Exploratory notebooks
│   └── phase1_demo.ipynb
├── scripts/
│   ├── run_scraper.py       # CLI: collect data
│   ├── run_preprocessing.py # CLI: clean corpus
│   └── run_baseline_demo.py # CLI: run BM25/TF-IDF search demo
├── requirements.txt
├── .env.example
└── README.md
```

## Phases

| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| 1 | 1–2 | Data Collection + Classical Baselines | ✅ In Progress |
| 2 | 3–4 | NER + Knowledge Graph (Neo4j) | 🔜 Planned |
| 3 | 5–6 | KG-RAG Integration + LLM | 🔜 Planned |
| 4 | 7–8 | Evaluation + Report + Demo | 🔜 Planned |

## Phase 1 Deliverables

- Multi-source scraper (TechCrunch, VentureBeat, YC, job boards)
- HTML cleaning + deduplication pipeline
- BM25 and TF-IDF retrieval baselines
- LDA topic modeling (10–15 topics)
- Corpus stored as structured JSON with provenance metadata

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/sbenstewart/cse573-semanticwebmining-project
cd cse573-semanticwebmining-project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your settings (Neo4j, API keys — needed in Phase 2+)
```

### 5. Run Phase 1 pipeline
```bash
# Step 1: Scrape data
python scripts/run_scraper.py --sources techcrunch venturebeat yc --max-articles 50

# Step 2: Clean and deduplicate
python scripts/run_preprocessing.py

# Step 3: Launch baseline search demo
python scripts/run_baseline_demo.py --query "LLM startup funding 2024"
```

## Data Schema

Each document in `data/processed/corpus.jsonl` follows this schema:

```json
{
  "doc_id": "tc_20240315_abc123",
  "source_url": "https://techcrunch.com/...",
  "publisher": "TechCrunch",
  "author": "Jane Doe",
  "published_date": "2024-03-15T10:30:00Z",
  "crawl_time": "2024-03-16T08:00:00Z",
  "raw_text": "...",
  "cleaned_text": "...",
  "title": "...",
  "tags": ["AI", "funding", "LLM"]
}
```

## Team

| Member | Role |
|--------|------|
| Ben Stewart Silas Sargunam | Data Collection Lead |
| Kelvin Panashe Munakandafa | Knowledge Graph Architect |
| Kumar Hasti | NLP & NER Engineer |
| Raksshitha Neelamegam Jothieswaran | RAG & LLM Integration |
| Shruthi Chandrakumar | Evaluation & UI Developer |
| Vibha Swaminathan | Evaluation & UI Developer |

## References

1. Robertson & Zaragoza (2009) — BM25
2. Blei, Ng & Jordan (2003) — LDA
3. Devlin et al. (2019) — BERT
4. Karpukhin et al. (2020) — DPR
5. Lewis et al. (2020) — RAG
6. Järvelin & Kekäläinen (2002) — nDCG
7. W3C PROV-O (2013)
8. W3C SHACL (2017)
