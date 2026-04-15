# Phase 4 Benchmark Report

_Source: `evaluation/results_20260415_123404.json`_

## Benchmark Run Metadata

- **Timestamp:** 2026-04-15T12:34:04.550300
- **Questions:** 25
- **Approaches benchmarked:** both
- **Total runtime:** 15.2 minutes (915s)

## Overall Comparison

| Metric | text_to_cypher | graph_rag |
|---|---|---|
| Questions | 25 | 25 |
| Answered (no error) | 24 | 25 |
| Answered rate | 96% | 100% |
| Mean keyword coverage | 63% | 67% |
| Mean latency | 21.6s | 15.0s |
| Median latency | 15.1s | 14.5s |
| Mean answer length (chars) | 152 | 371 |

## Per-Category Performance

### Keyword coverage by category

| Category | text_to_cypher | graph_rag |
|---|---|---|
| aggregation | 60% | 35% |
| factual_lookup | 73% | 80% |
| multi_hop | 60% | 40% |
| narrative | 60% | 100% |
| semantic | 60% | 80% |

### Answered rate by category

| Category | text_to_cypher | graph_rag |
|---|---|---|
| aggregation | 80% (4/5) | 100% (5/5) |
| factual_lookup | 100% (5/5) | 100% (5/5) |
| multi_hop | 100% (5/5) | 100% (5/5) |
| narrative | 100% (5/5) | 100% (5/5) |
| semantic | 100% (5/5) | 100% (5/5) |

### Mean latency by category

| Category | text_to_cypher | graph_rag |
|---|---|---|
| aggregation | 13.8s | 12.5s |
| factual_lookup | 45.6s | 13.5s |
| multi_hop | 18.1s | 15.7s |
| narrative | 20.0s | 17.4s |
| semantic | 10.3s | 16.1s |

## Per-Question Results

| Q | Category | Question | A: cov | A: status | B: cov | B: status |
|---|---|---|---|---|---|---|
| F1 | factual_lookup | Who invested in Replit? | 100% | ✓ | 100% | ✓ |
| F2 | factual_lookup | How much did Anthropic raise in its Series C? | 67% | ✓ | 100% | ✓ |
| F3 | factual_lookup | What is OpenAI's largest funding round in the graph? | 100% | ✓ | 100% | ✓ |
| F4 | factual_lookup | Which companies did Founders Fund invest in? | 0% | ✓ | 0% | ✓ |
| F5 | factual_lookup | What products has Scale AI announced? | 100% | ✓ | 100% | ✓ |
| A1 | aggregation | What are the 3 biggest funding rounds? | 0% | ✗ err | 50% | ✓ |
| A2 | aggregation | How many funding rounds are in the graph? | 100% | ✓ | 0% | ✓ |
| A3 | aggregation | Which investor appears in the most funding rounds? | 0% | ✓ | 0% | ✓ |
| A4 | aggregation | What is the total amount raised across all Series A rou... | 100% | ✓ | 100% | ✓ |
| A5 | aggregation | List all startups that have raised over $1 billion. | 100% | ✓ | 25% | ✓ |
| M1 | multi_hop | Which investors have funded both Anthropic and Replit? | 100% | ✓ | 100% | ✓ |
| M2 | multi_hop | What technologies are used by OpenAI's products? | 0% | ✓ | 0% | ✓ |
| M3 | multi_hop | Which companies use both RAG and vector databases? | 100% | ✓ | 0% | ✓ |
| M4 | multi_hop | Who are the investors in the biggest Series B round? | 0% | ✓ | 0% | ✓ |
| M5 | multi_hop | What products does Anthropic make, and what technologie... | 100% | ✓ | 100% | ✓ |
| S1 | semantic | What are AI coding startups saying about agent capabili... | 0% | ✓ | 100% | ✓ |
| S2 | semantic | What concerns have been raised about AI safety and alig... | 0% | ✓ | 0% | ✓ |
| S3 | semantic | Which startups are working on robotics or embodied AI? | 100% | ✓ | 100% | ✓ |
| S4 | semantic | What trends are visible in recent AI infrastructure inv... | 100% | ✓ | 100% | ✓ |
| S5 | semantic | Are there any controversies or incidents involving AI c... | 100% | ✓ | 100% | ✓ |
| N1 | narrative | Summarize recent activity in the AI coding assistant sp... | 100% | ✓ | 100% | ✓ |
| N2 | narrative | Tell me about NVIDIA's investments and partnerships in ... | 0% | ✓ | 100% | ✓ |
| N3 | narrative | What is the state of voice AI and speech recognition st... | 0% | ✓ | 100% | ✓ |
| N4 | narrative | Give me an overview of Anthropic's product line and fun... | 100% | ✓ | 100% | ✓ |
| N5 | narrative | What's happening with enterprise AI deployment accordin... | 100% | ✓ | 100% | ✓ |

## Key Findings

- Approach B had a higher answered rate (100% vs 96%), reflecting A's known difficulty generating valid aggregation Cypher.
- Mean keyword coverage was comparable across approaches (63% vs 67%).
- Mean latency: A 21.6s vs B 15.0s.
- Category-level divergence (>20% gap):
  - **aggregation:** A wins by 25% (A=60%, B=35%)
  - **narrative:** B wins by 40% (A=60%, B=100%)
  - **semantic:** B wins by 20% (A=60%, B=80%)

## Detailed Answers (per question)

### F1 [factual_lookup]
**Q:** Who invested in Replit?
_Expected keywords:_ Founders Fund, GV, Benchmark

**Approach A (text-to-Cypher)** (129.4s, keyword coverage 100%)
> Founders Fund, Tiger Global, Fidelity Management & Research Company LLC, Benchmark, and GV invested in Replit.

**Approach B (GraphRAG)** (20.8s, keyword coverage 100%)
> The investors mentioned in the knowledge graph are:
> 
> * Benchmark
> * Amazon
> * Founders Fund
> * Tiger Global
> * Fidelity Management & Research Company LLC
> * GV

### F2 [factual_lookup]
**Q:** How much did Anthropic raise in its Series C?
_Expected keywords:_ Anthropic, 1.5, Series C

**Approach A (text-to-Cypher)** (22.3s, keyword coverage 67%)
> Anthropic raised $1,500,000,000 in its Series C.

**Approach B (GraphRAG)** (18.1s, keyword coverage 100%)
> The query returned information about Anthropic's funding rounds, but this may not directly answer your question. However, according to the retrieved context, Anthropic raised $1.5B in its Series C funding round.

### F3 [factual_lookup]
**Q:** What is OpenAI's largest funding round in the graph?
_Expected keywords:_ OpenAI, 50

**Approach A (text-to-Cypher)** (32.1s, keyword coverage 100%)
> The query returned information about OpenAI's largest funding round, but this may not directly answer your question. The largest funding round for OpenAI in the graph is $50 billion, with no announced date provided.

**Approach B (GraphRAG)** (9.3s, keyword coverage 100%)
> The query returned facts about various startups, but this may not directly answer your question. However, according to the retrieved context, OpenAI's largest funding round is $50 billion (Growth).

### F4 [factual_lookup]
**Q:** Which companies did Founders Fund invest in?
_Expected keywords:_ Replit

**Approach A (text-to-Cypher)** (15.1s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (4.7s, keyword coverage 0%)
> I couldn't find that in the knowledge graph.

### F5 [factual_lookup]
**Q:** What products has Scale AI announced?
_Expected keywords:_ SGP

**Approach A (text-to-Cypher)** (29.0s, keyword coverage 100%)
> Scale AI has announced the following products:
> 
> * Claude 3.5 Sonnet
> * GenAI financial forecasting model
> * TIME's Person of the Year AI experience
> * Report generation for multiple customized report templates
> * Agentic warfare and scenario planning
> * Donovan
> * SGP
> * Scale Generative AI Platform
> * Replit Agent
> * Scale GenAI Platform
> * GenAI Platform
> * RLXF
> * Agent RL training platform
> * Agent RL Training Building algorithms
> * Agent building product
> * Scale's data engine
> * Agent Reinforcement Learning Environments data vertical
> * Computer Using Agent (CUA) data
> * Claude 3.5
> * GenAI Evaluation Suite
> * Scale Generative Platform (SGP)
> * Generative AI Platform
> * Scale Data Engine
> * Humanity's Last Exam
> * Remote Labor Index
> * SWE-Bench Pro
> * GRC function
> * GenAI Data Engine
> * Scale GenerativeAI Platform (SGP)
> * Generative AI Data Engine
> * Agent Development Platform
> * Scale Generative AI Platform (SGP)
> * SciPredict
> * PropensityBench
> * Professional Reasoning Benchmark
> * SGP (Scale Generative AI Platform)

**Approach B (GraphRAG)** (14.5s, keyword coverage 100%)
> Scale AI has announced the following products:
> 
> * Claude 3.5 Sonnet
> * GenAI financial forecasting model
> * TIME's Person of the Year AI experience
> * Report generation for multiple customized report templates
> * Agentic warfare and scenario planning
> * Donovan
> * SGP (Scale Generative AI Platform)
> * Replit Agent
> * Scale GenAI Platform
> * GenAI Platform
> * RLXF
> * Agent RL training platform

### A1 [aggregation]
**Q:** What are the 3 biggest funding rounds?
_Expected keywords:_ OpenAI, 50, Anthropic, 30

**Approach A (text-to-Cypher)** (19.6s, keyword coverage 0%)
> *ERROR:* `CypherSyntaxError: {code: Neo.ClientError.Statement.SyntaxError} {message: In a WITH/RETURN with DISTINCT or an aggregation, it is not possible to access variables declared before the WITH/RETURN: f (`

**Approach B (GraphRAG)** (13.1s, keyword coverage 50%)
> The query returned information about various funding rounds, but this may not directly answer your question. However, based on the available data:
> 
> * VC Eclipse has a $1.3B fund.
> * Antiverse raised $9.3M in Series A funding.
> * Oska Health received €11M in Seed funding.
> * Collide Capital raised a $95M fund.
> 
> Note that OpenAI's $50 billion growth round is mentioned, but it is not clear if this is the largest funding round overall.

### A2 [aggregation]
**Q:** How many funding rounds are in the graph?
_Expected keywords:_ 35

**Approach A (text-to-Cypher)** (6.8s, keyword coverage 100%)
> There are 35 funding rounds in the graph.

**Approach B (GraphRAG)** (13.6s, keyword coverage 0%)
> There are 7 funding rounds in the knowledge graph.
> 
> * Oska Health: €11M (Seed)
> * Replit: $20M (Series A)
> * Replit: $200M (Series C)
> * Cursor: $100M (Series A)
> * OpenAI: $50 billion (Growth)
> * VC Eclipse: $1.3B (Unknown)
> * Replit: $400M (Unknown)

### A3 [aggregation]
**Q:** Which investor appears in the most funding rounds?
_Expected keywords:_ Founders Fund

**Approach A (text-to-Cypher)** (14.8s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (6.8s, keyword coverage 0%)
> JPMorgan and Goldman Sachs both appear as investors, but only JPMorgan also appears as an investor for Collide Capital's $95M fund.

### A4 [aggregation]
**Q:** What is the total amount raised across all Series A rounds?
_Expected keywords:_ Series A

**Approach A (text-to-Cypher)** (8.4s, keyword coverage 100%)
> The total amount raised across all Series A rounds is $212,300,000.

**Approach B (GraphRAG)** (14.5s, keyword coverage 100%)
> The query returned information about various funding rounds, but this may not directly answer your question. However, I can provide some relevant facts.
> 
> Antiverse raised $9.3M in a Series A round.
> BrainCheck raised $13M in a Series A round.
> Cursor raised $100M in a Series A round.
> 
> I couldn't find the total amount raised across all Series A rounds.

### A5 [aggregation]
**Q:** List all startups that have raised over $1 billion.
_Expected keywords:_ OpenAI, Anthropic, Databricks, Sierra

**Approach A (text-to-Cypher)** (19.4s, keyword coverage 100%)
> Here is the list of startups that have raised over $1 billion:
> 
> * Coherent
> * Scale AI
> * Sierra
> * Anthropic
> * OpenAI
> * Commonwealth Fusion Systems
> * Databricks
> * VC Eclipse

**Approach B (GraphRAG)** (14.4s, keyword coverage 25%)
> Based on the retrieved context, the following startups have raised over $1 billion:
> 
> * Anthropic ($30 billion)
> * Replit ($400 million Series C and $200 million Series C)
> 
> Note: The query returned information about funding rounds, but this may not directly answer your question. However, based on the available data, these two startups have collectively raised over $1 billion.

### M1 [multi_hop]
**Q:** Which investors have funded both Anthropic and Replit?
_Expected keywords:_ Founders Fund

**Approach A (text-to-Cypher)** (23.8s, keyword coverage 100%)
> The query returned the names of investors who have funded both Anthropic and Replit. The two investors are Founders Fund and GV.

**Approach B (GraphRAG)** (20.1s, keyword coverage 100%)
> The query returned information about funding rounds for various startups, but this may not directly answer your question. However, based on the retrieved context, I found that the following investors have been mentioned as investing in both Anthropic and Replit:
> 
> * Palo Alto Networks
> * Apple
> * Microsoft
> * CrowdStrike
> * AWS
> * Cisco
> * Linux Foundation
> * OpenAI
> * GV
> * Founders Fund 
> 
> These investors are listed under the "Investors" section of the retrieved documents.

### M2 [multi_hop]
**Q:** What technologies are used by OpenAI's products?
_Expected keywords:_ transformer

**Approach A (text-to-Cypher)** (13.9s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (15.6s, keyword coverage 0%)
> The query returned information about various startups and their technologies, but this may not directly answer your question. However, according to the retrieved context, some of OpenAI's products use the following technologies:
> 
> * GPT-4o uses large language models.
> * ChatGPT Pro is a chatbot.
> 
> Note that the specific technologies used by each product are not explicitly stated in the retrieved context.

### M3 [multi_hop]
**Q:** Which companies use both RAG and vector databases?
_Expected keywords:_ Scale AI

**Approach A (text-to-Cypher)** (14.8s, keyword coverage 100%)
> The query returned a list of startups that use both RAG and vector databases. This may not directly answer your question, as it only provides the names of companies that meet this criterion.
> 
> Here is the list:
> 
> 1. Anthropic
> 2. Scale AI
> 3. AssemblyAI
> 4. OpenAI
> 5. Replit

**Approach B (GraphRAG)** (14.7s, keyword coverage 0%)
> The query returned information about the technologies used by Replit, but this may not directly answer your question. However, based on the retrieved context, Replit uses both RAG and a vector database.
> 
> Additionally, another source mentions that Google and Replit are expanding their partnership to bring vibe coding to more companies, which implies that Google is also using these technologies in some capacity.

### M4 [multi_hop]
**Q:** Who are the investors in the biggest Series B round?
_Expected keywords:_ Commonwealth Fusion

**Approach A (text-to-Cypher)** (24.4s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (6.9s, keyword coverage 0%)
> I couldn't find that in the knowledge graph. The query returned information about funding rounds and investors, but it does not mention a "biggest Series B round".

### M5 [multi_hop]
**Q:** What products does Anthropic make, and what technologies do they use?
_Expected keywords:_ Claude

**Approach A (text-to-Cypher)** (13.8s, keyword coverage 100%)
> Anthropic makes Claude Cowork, Claude Desktop, Claude Code, Claude 3.5 Sonnet, GPT-4o, and Claude 3.5. They use domain knowledge, transformer, RAG, vector database, terminal-based AI agent, large language models, and LangChain technologies.

**Approach B (GraphRAG)** (21.0s, keyword coverage 100%)
> Anthropic makes the following products:
> 
> * Claude Cowork
> * Claude Opus 4.6
> * Claude Mythos Preview
> * Claude Desktop
> * Claude Code
> * Anthropic AI model
> * Opus
> * Mythos
> * Claude 3.5 Sonnet
> * GPT-4o
> 
> Anthropic uses the following technologies:
> 
> * Domain knowledge
> * Transformer
> * RAG (Reactive Attention Generator)
> * Vector database
> * Terminal-based AI agent
> * LangChain
> * Large language models

### S1 [semantic]
**Q:** What are AI coding startups saying about agent capabilities?
_Expected keywords:_ agent

**Approach A (text-to-Cypher)** (16.3s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (21.9s, keyword coverage 100%)
> The query returned information about various AI startups and their products, but this may not directly answer your question. However, it mentions "agent capabilities" in the context of Replit Agent, a product mentioned in several sources. 
> 
> Replit Agent is listed as one of the products of Replit, along with other products like Replit V3 and Siraj. It's also mentioned alongside other AI coding assistants like Cursor IDE and Augment Code.
> 
> In terms of specific statements about agent capabilities, I couldn't find any direct quotes or descriptions in the retrieved documents.

### S2 [semantic]
**Q:** What concerns have been raised about AI safety and alignment?
_Expected keywords:_ safety

**Approach A (text-to-Cypher)** (5.6s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (9.5s, keyword coverage 0%)
> I couldn't find that in the knowledge graph.

### S3 [semantic]
**Q:** Which startups are working on robotics or embodied AI?

**Approach A (text-to-Cypher)** (14.4s, keyword coverage 100%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (11.0s, keyword coverage 100%)
> The query returned information about job openings at Scale AI, but this may not directly answer your question.
> 
> However, I found a mention of "physical AI" in the TechCrunch article, which might be related to robotics or embodied AI. According to the article, VC Eclipse has a new $1.3B fund to back startups working on physical AI.

### S4 [semantic]
**Q:** What trends are visible in recent AI infrastructure investments?

**Approach A (text-to-Cypher)** (7.9s, keyword coverage 100%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (13.7s, keyword coverage 100%)
> The query returned information about specific startups and investments, but this may not directly answer your question. However, some trends can be inferred from the retrieved context:
> 
> * NVIDIA invested $2 billion in Coherent to scale AI data center infrastructure.
> * SubcontractorHub secured an eight-figure growth investment to scale its AI-driven residential construction operating system.
> 
> These examples suggest that there is a trend of significant investments in AI infrastructure and startups working on AI-related technologies.

### S5 [semantic]
**Q:** Are there any controversies or incidents involving AI coding tools?

**Approach A (text-to-Cypher)** (7.3s, keyword coverage 100%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (24.3s, keyword coverage 100%)
> Yes, there is a reported incident of an AI-powered coding tool causing a "catastrophic failure" that wiped out a software company's database. The Replit CEO apologized for the incident, stating that their AI engine made a catastrophic error in judgment and destroyed all production data. 
> 
> This information comes from two news articles: "AI-powered coding tool wiped out a software company’s database in ‘catastrophic failure’" (Google News / Various, 2025-07-23T07:00:00+00:00) and "AI coding platform goes rogue during code freeze and deletes entire company database — Replit CEO apologizes after AI engine says it 'made a catastrophic error in judgment' and 'destroyed all producti" (Google News / Various, 2025-07-21T07:00:00+00:00).

### N1 [narrative]
**Q:** Summarize recent activity in the AI coding assistant space.
_Expected keywords:_ Replit

**Approach A (text-to-Cypher)** (13.7s, keyword coverage 100%)
> The query returned a list of investors in Replit's funding rounds, but this may not directly answer your question about recent activity in the AI coding assistant space. However, based on the provided information, here is what I found:
> 
> Replit has received investments from several notable firms, including Founders Fund, Tiger Global, Fidelity Management & Research Company LLC, Benchmark, and GV.

**Approach B (GraphRAG)** (16.9s, keyword coverage 100%)
> The query returned information about AssemblyAI and Replit, but this may not directly answer your question. 
> 
> AssemblyAI has released several products, including Conformer-1 API, a SOTA Speech Recognition Model (Source: "AssemblyAI releases Conformer-1 API, the SOTA Speech Recognition Model", Google News / Various). They have also developed other technologies such as vector database and transformer.
> 
> Replit has showcased no-code iOS app build capabilities using AI-driven development (Source: "Replit Showcases No-Code iOS App Build to Highlight AI-Driven Development Capabilities", Google News / Various).

### N2 [narrative]
**Q:** Tell me about NVIDIA's investments and partnerships in AI.
_Expected keywords:_ NVIDIA

**Approach A (text-to-Cypher)** (17.8s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (11.0s, keyword coverage 100%)
> NVIDIA has invested $2 billion in Coherent to scale AI data center infrastructure. They have also partnered with Lumentum to develop state-of-the-art optics technology. Additionally, they have invested $4 billion into photonics to scale AI data centers. 
> 
> Note: The query returned information about NVIDIA's investments and partnerships, but this may not directly answer your question as it focuses on specific areas of investment rather than a comprehensive overview of NVIDIA's efforts in AI.

### N3 [narrative]
**Q:** What is the state of voice AI and speech recognition startups?
_Expected keywords:_ AssemblyAI

**Approach A (text-to-Cypher)** (9.6s, keyword coverage 0%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (22.4s, keyword coverage 100%)
> The query returned information about AssemblyAI, a startup in the voice AI and speech recognition space. However, it does not directly answer your question about the state of voice AI and speech recognition startups.
> 
> That being said, based on the retrieved context, we can see that AssemblyAI has developed several products related to voice intelligence, including Claude 3.5 Sonnet, GPT-4o, Universal-Streaming, Conformer-1 API, Universal-1, and AssemblyAI Universal-1. These products utilize technologies such as vector databases, transformers, RAG, LangChain, and large language models.
> 
> It's worth noting that the funding rounds for AssemblyAI are listed as $115M (Unknown) and $28M (Unknown), but no specific information is provided about other voice AI or speech recognition startups.

### N4 [narrative]
**Q:** Give me an overview of Anthropic's product line and funding history.
_Expected keywords:_ Claude, Anthropic

**Approach A (text-to-Cypher)** (40.2s, keyword coverage 100%)
> Anthropic's product line includes Claude Cowork, Claude Opus 4.6, Claude Mythos Preview, Claude Desktop, Claude Code, Anthropic AI model, Opus, Mythos, Claude 3.5 Sonnet, and Cowork.
> 
> The query returned funding information for these products, but this may not directly answer your question about the product line. The funding rounds include Series C ($1.5B) and Unknown amounts of $30 billion, $1.5B, $1B, $200M, and $100M.

**Approach B (GraphRAG)** (23.1s, keyword coverage 100%)
> Anthropic has developed several products, including Claude Cowork, Claude Opus 4.6, Claude Mythos Preview, Claude Desktop, Claude Code, and the Anthropic AI model. They also have a product called Opus and another called Mythos.
> 
> As for funding, Anthropic has received various amounts from investors, including $200M (Unknown), $100M (Unknown), $30 billion (Growth), and $1.5B (Series C).

### N5 [narrative]
**Q:** What's happening with enterprise AI deployment according to the news?

**Approach A (text-to-Cypher)** (18.7s, keyword coverage 100%)
> I couldn't find that information in the knowledge graph.

**Approach B (GraphRAG)** (13.4s, keyword coverage 100%)
> Mirantis and Supermicro are accelerating sovereign AI and hybrid cloud deployments. Lenovo is scaling trusted AI-powered business computing through modular innovation and enterprise platforms. 
> 
> The query returned information about companies' innovations and partnerships, but this may not directly answer your question about what's happening with enterprise AI deployment according to the news.

