# TrendScout AI 2.0
**Knowledge Graph-Augmented Market Intelligence for AI Startups**

Arizona State University — CSE 573 Semantic Web Mining

## Project Overview
TrendScout AI 2.0 is a semantic web mining pipeline that monitors the AI startup ecosystem. It combines multi-source web mining, classical IR baselines, Knowledge Graph construction (Neo4j), and a KG-RAG conversational interface.

## Phases
| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| 1 | 1-2 | Data Collection + Classical Baselines | ✅ Complete |
| 2 | 3-4 | NER + Knowledge Graph (Neo4j) | 🔜 Planned |
| 3 | 5-6 | KG-RAG Integration + LLM | 🔜 Planned |
| 4 | 7-8 | Evaluation + Report + Demo | 🔜 Planned |

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
```

## Run Phase 1
```bash
python scripts/run_scraper.py --all --max-articles 50
python scripts/run_preprocessing.py
python scripts/run_baseline_demo.py
streamlit run scripts/streamlit_demo.py
pytest
```

## Team
| Member | Role |
|--------|------|
| Kelvin Panashe Munakandafa | Knowledge Graph Architect & Data Engineer — Ontology design, Neo4j modeling, Cypher queries, provenance representation, data pipeline infrastructure |
| _teammates — please add your name and role_ | |
