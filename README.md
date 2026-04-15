# TrendScout AI 2.0
**Knowledge Graph-Augmented Market Intelligence for AI Startups**

Arizona State University — CSE 573 Semantic Web Mining

## Project Overview
TrendScout AI 2.0 is a semantic web mining pipeline that monitors the AI startup ecosystem. It combines multi-source web mining, classical IR baselines, LLM-powered Knowledge Graph construction (Neo4j), and a dual KG-RAG conversational interface that answers natural-language questions about AI startups, funding rounds, and technology trends.

The system runs **fully locally** — text extraction and question-answering use a local Ollama model (`llama3.1:8b`), document embeddings use `BAAI/bge-base-en-v1.5` via sentence-transformers, the graph and vector index live in a local Dockerized Neo4j, and no external APIs are required.

## Phases
| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| 1 | 1-2 | Data Collection + Classical Baselines | ✅ Complete |
| 2 | 3-4 | LLM-based Knowledge Graph (Neo4j) | ✅ Complete |
| 3 | 5-6 | KG-RAG Integration + Natural-language Q&A | ✅ Complete |
| 4 | 7-8 | Evaluation + Report + Demo | 🔜 Planned |

---

## Setup

### Prerequisites
- Python 3.12+
- Docker (for Neo4j)
- [Ollama](https://ollama.com/download) (for local LLM)

### One-time install
```bash
python -m venv venv
source venv/bin/activate                    # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
cp .env.example .env
```

> **Note on Brotli:** the scraper sends `Accept-Encoding: gzip, deflate, br` to look like a real browser, which means some sources (notably VentureBeat) respond with Brotli-compressed HTML. The `brotli` package in `requirements.txt` is required for `requests` to auto-decompress these — without it, the scraper receives raw compressed bytes and silently drops the docs.

```bash# Pull the local LLM (Phase 2)
ollama pull llama3.1:8b

# Start Neo4j in Docker (Phase 2)
docker run -d \
  --name trendscout-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/trendscout123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_apoc_export_file_enabled=true \
  -v trendscout-neo4j-data:/data \
  neo4j:5.20

# Update .env with the Neo4j password (matches the docker-run command above)
sed -i '' 's/your_password_here/trendscout123/' .env       # macOS
# sed -i 's/your_password_here/trendscout123/' .env        # Linux
```

Verify everything is wired up:
```bash
python -m pytest -q                                         # should print "169 passed"
docker ps | grep trendscout-neo4j                           # container should be Up
ollama list | grep llama3.1                                 # model should be present
```

> **Note on Phase 3 dependencies:** `sentence-transformers` (which pulls in PyTorch) is required for the GraphRAG embedding pipeline. It's included in `requirements.txt`. On first run of `build_vector_index.py`, the `BAAI/bge-base-en-v1.5` model (~400 MB) is downloaded and cached in `~/.cache/huggingface`. PyTorch auto-detects Apple Silicon MPS on macOS for GPU-accelerated embedding.

---

## Phase 1 — Data Collection + Classical IR Baselines

**What it does:** scrapes AI-startup news and job postings from multiple sources, cleans and deduplicates the corpus, then builds BM25 and TF-IDF retrieval baselines for keyword search over the result.

### Run the Phase 1 pipeline
```bash
# 1. Get a corpus. Either scrape fresh:
python scripts/run_scraper.py --all --max-articles 50

#    OR convert the committed sample data (instant, no scraping):
python -c "
import json, jsonlines
from pathlib import Path
docs = json.load(open('data/master_corpus.json'))
Path('data/processed').mkdir(parents=True, exist_ok=True)
with jsonlines.open('data/processed/corpus.jsonl', 'w') as w:
    for d in docs: w.write(d)
print(f'Wrote {len(docs)} docs')
"

# 2. Clean + deduplicate
python scripts/run_preprocessing.py

# 3. Run the interactive search demo (BM25 + TF-IDF side by side)
python -m scripts.run_baseline_demo

# 4. Or the Streamlit UI
streamlit run scripts/streamlit_demo.py
```

### Phase 1 deliverables
- Multi-source scraper (TechCrunch, VentureBeat, YC, Google News, job boards)
- `TextCleaner` (HTML stripping, boilerplate removal, English filter, length filter)
- `Deduplicator` (MinHash LSH, Jaccard ≥ 0.85)
- `BM25Retriever` and `TFIDFRetriever` with `precision@k` / `nDCG@k`
- 62 unit tests covering scraper, preprocessing, retrieval
- Streamlit demo UI

---

## Phase 2 — LLM-based Knowledge Graph

**What it does:** runs a local LLM over each document in the cleaned corpus, extracts named-entity facts (funding rounds, products, technologies) as structured JSON, and writes them to Neo4j as a typed graph with full provenance back to source documents.

### Schema
```
Nodes:    Startup, Investor, FundingRound, Product, Technology,
          Skill, Event, Document
Edges:    HAS_FUNDING_ROUND, INVESTED_BY, ANNOUNCED, USES_TECH,
          MENTIONS, SOURCED_FROM, HIRING_FOR, PARTNERED_WITH
```

Every extracted fact links back to its source document via `SOURCED_FROM` and `MENTIONS` edges, enabling Phase 3 to cite sources when answering questions.

### Reproduce the KG from a clean Neo4j
```bash
# 1. Initialize the schema (idempotent; --wipe deletes existing data)
python scripts/run_kg_init.py --wipe

# 2. Pass 1: extract funding rounds (~20-40 min on M-series MacBook)
python scripts/run_kg_build.py --pass funding

# 3. Pass 2: extract products and technologies (~20-40 min)
python scripts/run_kg_build.py --pass products

# 4. Run surgical cleanup of obvious LLM extraction noise
python scripts/clean_kg.py --dry-run        # preview what would be deleted
python scripts/clean_kg.py                  # actually delete
```

### Inspect the graph
```bash
# Pretty-printed demo queries
python scripts/run_kg_queries.py            # all 11 canned queries
python scripts/run_kg_queries.py --list     # list available queries
python scripts/run_kg_queries.py --query top-tech

# Or open the Neo4j browser
open http://localhost:7474                  # login: neo4j / trendscout123
```

### Headline demo queries
- **`top-tech`** — most-mentioned technologies across all extracted products. Real ranking: RAG, LangChain, vector database, RLHF, transformer, PyTorch, LLMs.
- **`investor-portfolio`** — for each investor, the startups they funded. Multi-hop traversal demonstrating real co-investment patterns (Founders Fund → Augment Code, Replit, Anthropic).
- **`funded-and-shipping`** — startups that BOTH raised funding AND announced products. Cross-pass query joining Pass 1 + Pass 2 data.
- **`rounds-with-provenance`** — every funding round joined to its source article title and publisher, showcasing the provenance edges.

### Phase 2 deliverables
- 8-node-type schema with uniqueness constraints (`src/kg/schema.py`)
- Local-LLM-based `FundingExtractor` and `ProductTechExtractor` (Ollama, JSON-mode, retry-on-malformed)
- `FundingIngester` and `ProductTechIngester` with idempotent MERGE-based Cypher
- Name normalizer with seed alias tables for ~35 VC firms and ~10 startups (`src/kg/normalizer.py`)
- Deterministic post-extraction filter that catches "aggregate attribution" hallucinations (e.g. when an article reports a combined total across multiple companies)
- Surgical cleanup script (`scripts/clean_kg.py`) that deterministically removes obvious extraction noise
- 11 demo Cypher queries (`scripts/run_kg_queries.py`)
- 30 unit tests for the KG components, all using mocked Ollama (no live LLM in tests)

### Final graph state (after both passes + cleanup, 295-doc corpus)
| | Count |
|---|---|
| Documents | 295 |
| Startups | 58 |
| Investors | 51 |
| FundingRounds | 35 |
| Products | 106 |
| Technologies | 89 |
| Total nodes | ~579 |
| `HAS_FUNDING_ROUND` | 35 |
| `INVESTED_BY` | 61 |
| `ANNOUNCED` | 118 |
| `USES_TECH` | 157 |
| `MENTIONS` | 411 |
| `SOURCED_FROM` | 44 |

**Top funding rounds in the graph:** OpenAI $50B (Growth), Anthropic $30B (Growth), Databricks $20B, Sierra $10B, Coherent $2B (NVIDIA), Commonwealth Fusion Systems $1.8B (Series B), Scale AI $1.5B (Series F), Anthropic $1.5B Series C, VC Eclipse $1.3B. All extracted with correct source attribution via `SOURCED_FROM` edges to the originating TechCrunch / Google News articles.

**Top technologies by product-mention count:** RAG (15), transformer (12), vector database (9), LangChain (7), RLHF (6), PyTorch (4), LLMs (4).

### Corpus composition
| Publisher | Docs | Share |
|---|---|---|
| Scale AI (job postings) | 158 | 54% |
| Google News / Various | 77 | 26% |
| TechCrunch | 39 | 13% |
| VentureBeat | 14 | 5% |
| AssemblyAI (job postings) | 7 | 2% |
| **Total** | **295** | |

---

## Phase 3 — KG-RAG Question-Answering

Phase 3 adds a natural-language Q&A interface over the Phase 2 knowledge graph. Two approaches are implemented and benchmarked side-by-side:

**Approach A — Text-to-Cypher (classical baseline).** An LLM translates the user's question into a Cypher query, which is validated by a read-only safety layer, executed against Neo4j, and the results are formatted into an English answer with source citations. Best for structured factual lookups ("Who invested in Replit?", "Which companies use RAG?").

**Approach B — GraphRAG (main contribution).** The user's question is embedded using `BAAI/bge-base-en-v1.5`, the top-5 most similar documents are retrieved from a Neo4j native vector index (768-dim, cosine similarity), each document's KG neighborhood is expanded (startups, funding rounds, investors, products, technologies), and the combined textual + structured context is fed to the LLM for answer generation. Handles both structured and semantic queries ("What are startups saying about agent frameworks?").

Both approaches implement the same `BaseQASystem` protocol and return standardized `Answer` objects, enabling head-to-head benchmarking in Phase 4.

### One-time setup: build the vector index

```bash
# Downloads bge-base-en-v1.5 (~400 MB) on first run, then cached.
# Embeds 295 documents and stores vectors on Neo4j Document nodes.
python scripts/build_vector_index.py --wipe
```

Typical runtime: ~30-60 seconds on Apple Silicon (MPS-accelerated).

### Run the Q&A chat

```bash
# Interactive REPL — both approaches side-by-side (default)
python scripts/run_qa_chat.py

# Single approach
python scripts/run_qa_chat.py --approach A
python scripts/run_qa_chat.py --approach B

# One-shot mode
python scripts/run_qa_chat.py --query "Who invested in Replit?"
python scripts/run_qa_chat.py --approach B --query "What are the biggest funding rounds?"
```

### Example output

```
> Who invested in Replit?

  [TEXT TO CYPHER]
  Founders Fund, Tiger Global, Fidelity Management & Research Company
  LLC, Benchmark, and GV invested in Replit.
  (18761ms)

  [GRAPH RAG]
  Benchmark, Founders Fund, Tiger Global, Fidelity Management & Research
  Company LLC, GV.
  (15384ms)
  Sources: 5 document(s)
```

### Phase 3 architecture

```
User question
    │
    ├── Approach A: Text-to-Cypher
    │   ├── CypherGenerator (LLM + few-shot prompt + SCHEMA_PROMPT)
    │   ├── SafeCypherExecutor (read-only validator → Neo4j)
    │   ├── retry once on CypherSyntaxError (feed error back to LLM)
    │   └── AnswerFormatter (LLM + ANSWER_STYLE_GUIDE)
    │
    └── Approach B: GraphRAG
        ├── Embedder (bge-base-en-v1.5, 768-dim)
        ├── VectorStore.query (Neo4j native vector index, top-5)
        ├── Graph expansion (MENTIONS → Startup → FundingRound/Investor/Product/Tech)
        └── LLM answer generation (structured context + style guide)
```

### Safety layer

All LLM-generated Cypher passes through `cypher_safety.validate_read_only()` before execution. Two-layer defense: (1) first-clause whitelist (`MATCH`, `RETURN`, `WITH`, `UNWIND`, `OPTIONAL`), (2) forbidden-keyword scan (16 keywords: `CREATE`, `MERGE`, `DELETE`, `SET`, `DROP`, `CALL`, etc.). String literals and comments are stripped before scanning to avoid false positives. Any destructive query is rejected before it reaches Neo4j.

### Known tradeoffs between approaches

| Query type | Approach A | Approach B |
|---|---|---|
| Structured lookup ("Who invested in X?") | ✅ Precise Cypher traversal | ✅ Works via graph expansion |
| Aggregation ("Top 3 funding rounds") | ❌ 8B model struggles with Cypher aggregation | ⚠️ Partial — finds relevant docs but ranking is approximate |
| Multi-hop ("Investors who co-invest in LLM companies") | ✅ Natural fit for Cypher | ✅ Graph expansion covers this |
| Semantic ("What are startups saying about agents?") | ❌ No text retrieval capability | ✅ Vector search finds relevant articles |
| Latency (typical) | 8-20s (2 LLM calls) | 15-20s (1 embedding + 1 LLM call + graph queries) |

### Known limitations (honest)
1. **No `MAKES` vs `USES` distinction.** The LLM extracts both "Anthropic makes Claude" and "Replit uses Claude" under the same `ANNOUNCED` edge. Future work: split into `MAKES_PRODUCT` vs `USES_PRODUCT`.
2. **Surface-form variants persist.** ~5 variants of "Scale Generative AI Platform" (SGP, Scale GP, Scale Generative Platform, etc.) live as distinct Product nodes. Future work: extend the normalizer alias table.
3. **Funding precision ~80%** on the current corpus. The bulk of the news content is Google News headline-and-preview snippets, which give the LLM thin context. The TechCrunch and VentureBeat additions help meaningfully but the corpus still contains ~77 preview-only Google News docs.
4. **Corpus skew.** The corpus is 54% Scale AI job postings (158/295 docs). Pass 2 product extraction is dominated by Scale AI's internal product names. The TechCrunch and VentureBeat additions reduced this from the original 65% but a fully rebalanced corpus would require either YC/HackerNews scraping (currently broken — YC's directory is a JS-rendered SPA) or additional news sources.
5. **YC scraper not functional.** Y Combinator's company directory is a Next.js SPA that renders the company list client-side; the current `BeautifulSoup`-based scraper sees only the empty React shell. Fixing this would require either Playwright (already in `requirements.txt`) or hitting YC's underlying Algolia API directly. Listed in `src/scraper/yc_scraper.py` as future work.
6. **Text-to-Cypher struggles with aggregation queries.** The local llama3.1:8b model frequently generates syntactically invalid Cypher for queries involving `ORDER BY ... LIMIT` with `DISTINCT` or `count()`. A larger model (e.g. 70B) or chain-of-thought prompting would likely fix this. The retry mechanism catches the syntax error gracefully but cannot always self-correct.
7. **GraphRAG retrieval doesn't capture ranking/superlative semantics.** "What are the *biggest* funding rounds?" retrieves articles *about* funding but not specifically the *largest* ones, because cosine similarity doesn't encode ordinal concepts. A hybrid approach combining vector search with a structured Cypher pre-filter (`WHERE r.amount_usd > threshold`) would address this.
8. **Document text not stored in Neo4j.** The vector index embeds `title + first 2000 chars`, but the full `cleaned_text` is only in `corpus.jsonl`. GraphRAG's context blocks rely on graph-expanded structured facts rather than full article text, which limits its ability to answer questions requiring deep reading of article content.

---

## Scraping ethics

This project scrapes publicly accessible news articles for academic research purposes in compliance with each source's `robots.txt`. The scraper:

- **Respects `robots.txt`** rules at fetch time (`src/scraper/base_scraper.py`)
- **Rate-limits requests** to 2 seconds between fetches per source (`SCRAPER_RATE_LIMIT_SECONDS`)
- **Uses a standard browser User-Agent** because several sources rate-limit bot-style identifiers; this is consistent with common practice in academic NLP research and with the principles in *hiQ Labs v. LinkedIn* (9th Cir. 2022) which held that scraping public web data does not violate the CFAA. The User-Agent is configurable via the `SCRAPER_USER_AGENT` environment variable.
- **Cites every extracted fact** back to its source document via `SOURCED_FROM` and `MENTIONS` edges in the knowledge graph
- **Does not** bypass paywalls, login walls, or authentication of any kind
- **Does not** redistribute scraped content; the corpus is used solely for fact extraction and analysis, and the original source URLs are preserved as provenance

---

## Project layout
```
trendscout-ai/
├── config/
│   └── settings.py                 # env-driven configuration
├── data/
│   ├── master_corpus.json          # committed sample corpus (414 docs)
│   └── processed/corpus.jsonl      # Phase 1 output, Phase 2+3 input
├── backups/
│   └── kg_v3_backup.cypher         # canonical Phase 2 graph snapshot
├── src/
│   ├── corpus.py                   # corpus I/O + stats
│   ├── preprocessing/              # Phase 1: cleaner, deduplicator
│   ├── retrieval/                  # Phase 1: BM25, TF-IDF
│   ├── scraper/                    # Phase 1: source scrapers
│   ├── baseline/                   # Phase 1: LDA topic model
│   ├── kg/                         # Phase 2: KG construction
│   │   ├── schema.py               # node/edge schema + constraints
│   │   ├── neo4j_client.py         # driver wrapper with retries
│   │   ├── normalizer.py           # name canonicalization + aliases
│   │   ├── extractor.py            # LLM extractors (funding, products)
│   │   └── ingester.py             # Cypher MERGE-based writers
│   └── rag/                        # Phase 3: KG-RAG Q&A
│       ├── common.py               # Answer dataclass, BaseQASystem protocol
│       ├── cypher_safety.py        # read-only Cypher validator
│       ├── cypher_executor.py      # safe executor wrapping Neo4jClient
│       ├── cypher_generator.py     # LLM text-to-Cypher with few-shot
│       ├── answer_formatter.py     # LLM result-to-English formatter
│       ├── text_to_cypher.py       # Approach A pipeline
│       ├── embedder.py             # bge-base-en-v1.5 wrapper
│       ├── vector_store.py         # Neo4j native vector index ops
│       └── graph_rag.py            # Approach B pipeline
├── scripts/
│   ├── run_scraper.py              # Phase 1
│   ├── run_preprocessing.py        # Phase 1
│   ├── run_baseline_demo.py        # Phase 1
│   ├── streamlit_demo.py           # Phase 1
│   ├── run_kg_init.py              # Phase 2 — schema setup
│   ├── run_kg_build.py             # Phase 2 — extraction passes
│   ├── clean_kg.py                 # Phase 2 — surgical cleanup
│   ├── run_kg_queries.py           # Phase 2 — demo queries
│   ├── build_vector_index.py       # Phase 3 — embed docs into Neo4j
│   └── run_qa_chat.py              # Phase 3 — interactive Q&A REPL
└── tests/
    ├── test_scraper.py
    ├── test_preprocessing.py
    ├── test_retrieval.py
    ├── test_kg_extractor.py
    ├── test_rag_step1.py           # safety + common types
    ├── test_rag_step2.py           # executor + generator + formatter
    ├── test_rag_step3.py           # embedder + vector store
    └── test_rag_step4.py           # GraphRAG pipeline
```

## Tests
```bash
python -m pytest -q             # 169 tests, all passing
```

## Team
| Member | Role |
|--------|------|
| Kelvin Panashe Munakandafa | Knowledge Graph Architect & Data Engineer — Ontology design, Neo4j modeling, Cypher queries, provenance representation, data pipeline infrastructure |
| _teammates — please add your name and role_ | |
