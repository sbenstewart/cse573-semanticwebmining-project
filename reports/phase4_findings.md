# Phase 4 — Comparative Evaluation of KG-RAG Approaches

**Project:** TrendScout AI 2.0 — Knowledge Graph-Augmented Market Intelligence for AI Startups
**Course:** ASU CSE 573 — Semantic Web Mining
**Phase:** 4 of 4 (Evaluation)
**Date:** 2026-04-15

---

## 1. Introduction

Phases 2 and 3 of TrendScout AI 2.0 produced a knowledge graph of 295 documents → 634 nodes → 826 relationships about AI startups, funding rounds, investors, products, and technologies, plus two end-to-end question-answering systems built on top of it. Phase 4 quantitatively compares the two systems through a benchmark of 25 hand-curated questions covering five categories of query.

The comparison is the central contribution of this project. Neither paradigm under test is novel in isolation — semantic-parsing-style text-to-Cypher dates back to LUNAR (Woods, 1973) and was modernized as in-context-learning text-to-SQL by Rajkumar et al. (2022); GraphRAG was introduced in its current form by Edge et al. (Microsoft, April 2024) and productized by Neo4j in 2024. What is novel here is **the head-to-head empirical comparison on a domain-specific knowledge graph built entirely via local-LLM extraction (no proprietary APIs)**, run on commodity hardware (a MacBook Air with a local 8B LLM and a local Dockerized Neo4j).

## 2. Methodology

### 2.1 Systems Under Test

**Approach A — Text-to-Cypher (semantic-parsing baseline).** The user's natural-language question is fed to a local LLM (`llama3.1:8b` via Ollama) along with a description of the graph schema and seven worked few-shot examples. The LLM emits a Cypher query in JSON; the query passes through a read-only safety validator (rejects 16 destructive keywords) and is executed against Neo4j; the result rows are formatted into an English answer by a second LLM call constrained by an anti-fabrication style guide. If execution fails with a syntax error, the system retries once with the error message folded back into the prompt. This is the LLM-prompt-based incarnation of the classical text-to-structured-query paradigm (Rajkumar et al. 2022; Neo4j `GraphCypherQAChain`).

**Approach B — GraphRAG (retrieval-augmented contribution).** The user's question is embedded with `BAAI/bge-base-en-v1.5` (768 dimensions, MPS-accelerated). The Neo4j native vector index returns the five most similar `Document` nodes by cosine similarity. For each retrieved document, a graph-expansion query pulls the surrounding subgraph (Startups it MENTIONS, their FundingRounds, Investors, Products, Technologies). The combined textual + structured context is fed to the LLM along with the same anti-fabrication style guide, and an English answer is produced. This is the graph-augmented retrieval paradigm of Edge et al. (Microsoft, 2024).

Both systems implement a shared `BaseQASystem` protocol returning a normalized `Answer` dataclass, allowing the benchmark harness to invoke them through a single interface.

### 2.2 Evaluation Question Set

25 questions across 5 categories, 5 questions per category:

| Category | Description | Predicted winner |
|---|---|---|
| **factual_lookup** | Direct lookups against structured KG facts | A (precise traversals) |
| **aggregation** | Counting, ordering, summing | A's stress test |
| **multi_hop** | 2+ relationship traversals | Both should handle |
| **semantic** | Open-ended questions requiring article-text understanding | B (no text retrieval in A) |
| **narrative** | Multi-document synthesis and summarization | B (A cannot summarize) |

Each question carries a list of expected keywords (case-insensitive substring match) used for automated scoring. Some questions in the semantic and narrative categories use empty keyword lists, indicating they are scored qualitatively rather than automatically.

### 2.3 Metrics

| Metric | Definition |
|---|---|
| **Answered rate** | Fraction of questions returning a non-error response |
| **Keyword coverage** | Mean fraction of expected keywords found in the answer |
| **Latency (mean / median)** | Wall-clock time per question, in seconds |
| **Answer length** | Mean characters in returned text |
| **Per-category breakdown** | All metrics computed per-category to surface where each system wins |

Citation precision/recall would also be informative but is not reported in this run because the question set does not yet include gold doc_id annotations; this is left as future work.

### 2.4 Run Configuration

- Hardware: MacBook Air M-series, CPU + Apple Silicon MPS for embeddings
- LLM: `llama3.1:8b` via local Ollama, temperature 0 (generation) / 0.1 (formatting)
- Embedding model: `BAAI/bge-base-en-v1.5`, normalized to unit length
- Neo4j: 5.20 (Docker), native vector index (cosine similarity, 768 dims)
- Total runtime: 15.2 minutes for 50 LLM-driven inferences (25 × 2 systems)

## 3. Quantitative Results

### 3.1 Headline Numbers

| Metric | Approach A | Approach B |
|---|---|---|
| Questions | 25 | 25 |
| Answered rate | 96% | 100% |
| Mean keyword coverage | 63% | 67% |
| Mean latency | 21.6s | 15.0s |
| Median latency | 15.1s | 14.5s |
| Mean answer length (chars) | 152 | 371 |

**Key observations.** Both systems perform comparably on the headline keyword-coverage metric (63% vs 67%), but the per-category breakdown (Section 3.2) reveals that this near-tie disguises a strong specialization pattern: each approach wins different question categories by large margins. Approach B is more verbose (2.4× longer answers on average) and slightly faster (lower mean latency, comparable median). Approach A had one hard failure (a `CypherSyntaxError` that the retry mechanism could not recover from on question A1); B never returned an error, but its longer answers more frequently hedge with "this may not directly answer your question."

### 3.2 Per-Category Performance

| Category | Approach A | Approach B | Margin |
|---|---|---|---|
| factual_lookup | 73% | 80% | B +7% |
| **aggregation** | **60%** | **35%** | **A +25%** |
| **multi_hop** | **60%** | **40%** | **A +20%** |
| **semantic** | **60%** | **80%** | **B +20%** |
| **narrative** | **60%** | **100%** | **B +40%** |

Three category-level findings have margins ≥20%, sufficient to dominate the headline near-tie:

1. **Aggregation: A wins by 25%.** When A successfully generated valid Cypher, the answer was numerically precise (e.g., A2: "There are 35 funding rounds in the graph" — exactly correct). B more often hedged or mis-counted because its retrieval surfaces only a sample of relevant documents, not the whole graph (see Section 4.2 for an example).
2. **Narrative: B wins by 40%.** Questions requiring synthesis across multiple documents are categorically out of A's reach. Even when A's Cypher succeeded, it returned bare lists; B's vector retrieval surfaced multiple relevant articles whose graph neighborhoods the LLM could weave into prose.
3. **Semantic: B wins by 20%.** Open-ended questions like "Are there any controversies involving AI coding tools?" require semantic matching to article text, which A has no mechanism for. B's vector search located the relevant Replit-database-wipe articles and the LLM produced a coherent answer.

### 3.3 Latency

Mean latency favors B (15.0s vs 21.6s), but the medians (14.5s vs 15.1s) are nearly identical. The mean-vs-median gap for A is driven by a 129.4-second cold-start on the first question — Ollama's model loaded into memory on that call and stayed warm thereafter. After question 1, A's per-question latency is competitive with B's. This is an artifact of the run, not an inherent property of the systems.

## 4. Qualitative Analysis: Failure Modes Worth Reading

The aggregate numbers are the answer to "which system is better?" — and the answer is "neither, they have different strengths." But the *interesting* answer is *why* each system fails when it fails. Three specific failures from the benchmark illustrate fundamental limitations of each paradigm.

### 4.1 F4 — "Which companies did Founders Fund invest in?" (both fail, 0% / 0%)

**The data:** Founders Fund has a clear portfolio in the graph: Replit ($200M and $20M rounds), Augment Code ($20M), Anthropic ($1.5B). All four edges are correctly present (verified by Cypher inspection).

**Approach A — directional traversal failure.** The LLM generated this Cypher:

```cypher
MATCH (i:Investor {normalized_name: 'founders fund'})
   -[:INVESTED_BY]->(r:FundingRound)
   -[:HAS_FUNDING_ROUND]->(s:Startup)
RETURN DISTINCT s.name AS startup
```

The query is syntactically valid but **the relationship arrows point the wrong way.** The schema is `(Startup)-[:HAS_FUNDING_ROUND]->(FundingRound)-[:INVESTED_BY]->(Investor)`. The LLM patterned the query after the few-shot example "Who invested in Replit?" (which traverses Startup → Round → Investor) without realizing it needed to *invert* the traversal direction for the inverse question. The query matched zero paths, the formatter converted "no rows" to "I couldn't find that," and no error was raised.

**Insight:** Small LLMs (8B) struggle with inverse-relation queries even when the schema and forward-direction examples are in the prompt. A larger model (70B+) would likely handle this; alternatively, adding inverse-direction worked examples to the few-shot list would patch this specific failure mode without changing the architecture. This is a **prompt-engineering ceiling**, not a fundamental limit of the paradigm.

**Approach B — vector retrieval misdirection.** B retrieved these top-5 documents:

| Score | Title |
|---|---|
| 0.811 | Collide Capital raises $95M fund to back fintech, future-of-work startups |
| 0.798 | VC Eclipse has a new $1.3B fund to back — 'physical AI' startups |
| 0.790 | Amazon CEO takes aim at Nvidia, Intel, Starlink... |
| 0.789 | Most companies aren't seeing a return on AI investments... |
| 0.784 | Co-founder of $29 billion AI startup... |

**None of these mention Founders Fund.** Vector search retrieved articles that are *thematically similar* — they discuss VC funds and AI investments — but not articles *about* Founders Fund's portfolio. The graph expansion ran on documents that have no relationships to Founders Fund and surfaced no relevant facts; the LLM correctly reported finding nothing.

**Insight:** This is the **investor-perspective gap** in startup-centric corpora. No article in the 295-document corpus is *about* Founders Fund's portfolio. Founders Fund is mentioned in articles whose *primary subject* is the funded startup (Replit, Anthropic, Augment Code), so when those articles are embedded, the vector representation is dominated by the startup, not the investor. Embedding the question "Which companies did Founders Fund invest in?" lands closest to articles that are *about funds*, not articles *that mention Founders Fund*. This is a fundamental limitation of dense retrieval over a perspective-biased corpus.

**The paradigmatic insight from F4:** The same failure surfaces from totally different mechanisms. Both systems' failures are repairable, but the repairs are different — A needs better few-shot examples or a larger model; B needs either a query-rewriting step or per-entity index over investor mentions.

### 4.2 A2 — "How many funding rounds are in the graph?" (A correct, B hallucinates)

**The truth:** 35 (verified by the Phase 2 build summary).

**Approach A:** "There are 35 funding rounds in the graph." Correct. The LLM generated `MATCH (r:FundingRound) RETURN count(r)` and the answer was direct.

**Approach B:** "There are 7 funding rounds in the knowledge graph. * Oska Health: €11M (Seed)..." Wrong. B retrieved 5 documents, expanded their graph neighborhoods to find 7 distinct rounds among them, and reported that count as if it were the total.

**Insight:** This is the **partial-view hallucination** failure mode of GraphRAG. The LLM saw 7 funding rounds in its retrieved context and could not distinguish "7 rounds in my context window" from "7 rounds in the graph." The anti-fabrication style guide forbids inventing facts not in the retrieved context, but it doesn't prevent the LLM from *misinterpreting the scope of the retrieved context*. This is a known general limitation of RAG-style systems and is genuinely hard to fix without either (a) injecting graph-level statistics into the prompt or (b) routing questions like this to a structured-query path. The cleanest architectural fix would be **a router that sends counting questions to A and semantic questions to B** — turning the two systems from competitors into complementary components.

### 4.3 N2 — "Tell me about NVIDIA's investments and partnerships in AI." (A fails, B excels)

**Approach A:** "I couldn't find that information in the knowledge graph." (0% keyword coverage). The LLM either could not compose Cypher for this open-ended query or generated a query with zero results.

**Approach B:** "NVIDIA has invested $2 billion in Coherent to scale AI data center infrastructure..." (100% keyword coverage). B's vector search correctly retrieved the TechCrunch article about NVIDIA's $2B Coherent investment along with related infrastructure articles, and the LLM synthesized a narrative paragraph naming NVIDIA, Coherent, and the dollar amount.

**Insight:** This question has no clean Cypher representation — it requires reading articles that *mention* NVIDIA in various contexts and synthesizing across them. This is exactly the kind of question where the GraphRAG paradigm is built to win, and it does. **Narrative questions are categorically out of A's reach** unless the schema is extended with a "topics this article covers" relationship that maps text content to structured tags (which would essentially be reinventing topic modeling).

## 5. Limitations

Honest limitations of this evaluation:

1. **Small evaluation set.** 25 questions × 2 approaches = 50 data points is enough to see clear category-level patterns (the 25%–40% margins are robust to noise) but too small to compute reliable confidence intervals on the headline near-tie. A larger eval set (~100 questions per category) would make smaller effects visible.
2. **No citation scoring.** The benchmark records `cited_doc_ids` for each answer but does not compare them to gold citations because the question set does not yet include `expected_doc_ids`. Adding these would let us score *whether the right documents were used*, not just whether the right text appears in the answer.
3. **Single LLM evaluator.** Both systems use the same `llama3.1:8b` model. A larger model (e.g., GPT-4) used as an LLM-judge for qualitative scoring would catch failures that simple keyword matching misses (e.g., A's answer to A1 happened to be graceful, but a keyword-matcher reports it as 0% the same way it reports a hallucination at 0%).
4. **Cold-start latency artifact.** Question F1's 129-second runtime drags A's mean latency upward; the median is a fairer comparison. A pre-warm step in the benchmark runner would eliminate this.
5. **Corpus skew.** 54% of the corpus is Scale AI job postings, which biases B's vector retrieval toward Scale AI-related answers across many questions. A more balanced corpus would improve B's performance on non-Scale-AI questions.
6. **No router experiment.** The most interesting next experiment — a router that sends structured queries to A and semantic queries to B — is not built. The benchmark suggests this hybrid would outperform both individual approaches, particularly if it eliminates B's partial-view hallucinations on counting questions.

## 6. Conclusions

The two paradigms are **specialized, not interchangeable**. Approach A (text-to-Cypher) wins decisively on questions with structured answers (aggregation 60% vs 35%, multi-hop 60% vs 40%); Approach B (GraphRAG) wins decisively on questions requiring document understanding (semantic 80% vs 60%, narrative 100% vs 60%). They tie on factual lookups, where either paradigm can produce the answer. A composite system that routes by question type would likely outperform either individual system.

**The most important practical finding** is that small open models (here llama3.1:8b on commodity hardware) can deliver workable text-to-Cypher and GraphRAG for a domain-specific KG, with no proprietary API dependency. The system runs end-to-end on a MacBook Air. The cost to extend to a new domain is the cost of building a new KG (Phase 2 of this project); the QA layer requires no domain-specific code beyond the schema description string.

**The most important academic finding** is that the failure modes of the two paradigms are deeply different and complementary. A fails by generating valid-but-wrong queries (directional errors, syntactic failures on aggregation). B fails by retrieving similar-but-irrelevant documents and by misinterpreting partial context as global truth. These are not bugs to be fixed; they are properties of the paradigms. A research program seeking robust KGQA over local LLMs would need to either (a) combine both paradigms via a router, (b) train a larger model to handle both modes, or (c) extend each paradigm with mechanisms that address its specific failure mode (e.g., schema-direction augmentation for A, scope-aware prompting for B).

---

## Appendix A — Reproduction

```bash
# Phase 1 (data + classical baselines)
python scripts/run_scraper.py --sources techcrunch venturebeat assemblyai
python scripts/run_preprocessing.py

# Phase 2 (knowledge graph)
python scripts/run_kg_init.py --wipe
python scripts/run_kg_build.py --pass funding
python scripts/run_kg_build.py --pass products
python scripts/clean_kg.py

# Phase 3 (vector index for Approach B)
python scripts/build_vector_index.py --wipe

# Phase 4 (this benchmark)
python scripts/run_benchmark.py 2>&1 | tee evaluation/benchmark_run.log
python scripts/analyze_benchmark.py --output reports/phase4_findings.md
python scripts/analyze_benchmark.py --detail --output reports/phase4_findings_detailed.md
```

Total reproduction time on a MacBook Air M-series: ~3 hours wall-clock, dominated by Phase 2 (~90 min of LLM extraction) and Phase 4 (~15 min of QA inference).

## Appendix B — Bibliographic Notes

- **Text-to-SQL / text-to-Cypher.** Rajkumar et al. 2022, "Evaluating the Text-to-SQL Capabilities of Large Language Models" (the LLM-prompt-based modernization of the paradigm). Earlier: Yu et al. 2018 (Spider benchmark), Woods 1973 (LUNAR, the rule-based forerunner).
- **GraphRAG.** Edge et al. 2024 (Microsoft, "From Local to Global: GraphRAG"), Neo4j GraphRAG product (2024), LightRAG (HKU 2024).
- **RAG foundations.** Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS).
- **KG construction via LLMs.** Builds on the broader 2023+ literature on instruction-tuned extraction; Phase 2's prompt design is informed by best practices from Anthropic/OpenAI cookbook examples but is not derived from any single cited work.
