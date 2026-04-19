"""Approach B: GraphRAG — vector retrieval + graph expansion + LLM answer.

Pipeline:
  1. Embed the user's question using the same model as the index.
  2. Query the Neo4j vector index for top-K similar documents.
  3. For each retrieved document, expand its KG neighborhood:
       - Startups it MENTIONS
       - Their funding rounds, investors, products, technologies
  4. Assemble the document text + structured graph context into a prompt.
  5. Ask the LLM to answer the question using ONLY that context.

Implements BaseQASystem so Phase 4 can benchmark it alongside Approach A.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import ollama

from src.kg.neo4j_client import Neo4jClient
from src.rag.common import ANSWER_STYLE_GUIDE, SCHEMA_PROMPT, Answer
from src.rag.answer_formatter import extract_cited_doc_ids
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph expansion queries
# ---------------------------------------------------------------------------

_EXPAND_NEIGHBORHOOD = """\
MATCH (d:Document {doc_id: $doc_id})
OPTIONAL MATCH (d)-[:MENTIONS]->(s:Startup)
OPTIONAL MATCH (s)-[:HAS_FUNDING_ROUND]->(r:FundingRound)
OPTIONAL MATCH (r)-[:INVESTED_BY]->(i:Investor)
OPTIONAL MATCH (s)-[:ANNOUNCED]->(p:Product)
OPTIONAL MATCH (p)-[:USES_TECH]->(t:Technology)
RETURN d.title AS doc_title,
       d.publisher AS doc_publisher,
       d.published_date AS doc_date,
       collect(DISTINCT s.name) AS startups,
       collect(DISTINCT {
           amount: r.amount_raw,
           amount_usd: r.amount_usd,
           round_type: r.round_type,
           startup: s.name
       }) AS funding_rounds,
       collect(DISTINCT i.name) AS investors,
       collect(DISTINCT p.name) AS products,
       collect(DISTINCT t.name) AS technologies
"""


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _build_context_block(
    doc: dict,
    text_snippet: str,
    graph_neighborhood: dict,
) -> str:
    """Format one retrieved document + its graph neighborhood as context."""
    lines = [
        f"--- Source: {doc.get('title', 'Untitled')} "
        f"({doc.get('publisher', '?')}, {doc.get('published_date', '?')}) ---",
    ]

    # Text snippet (first ~1500 chars to save prompt space)
    if text_snippet:
        lines.append(f"Text: {text_snippet[:1500]}")

    gn = graph_neighborhood
    if gn.get("startups"):
        lines.append(f"Startups mentioned: {', '.join(gn['startups'])}")
    if gn.get("investors"):
        investors = [i for i in gn["investors"] if i]
        if investors:
            lines.append(f"Investors: {', '.join(investors)}")
    if gn.get("funding_rounds"):
        rounds = [
            r for r in gn["funding_rounds"]
            if r.get("amount") and r.get("startup")
        ]
        if rounds:
            round_strs = [
                f"{r['startup']}: {r['amount']} ({r.get('round_type', '?')})"
                for r in rounds
            ]
            # Deduplicate
            seen = set()
            unique = []
            for s in round_strs:
                if s not in seen:
                    unique.append(s)
                    seen.add(s)
            lines.append(f"Funding rounds: {'; '.join(unique)}")
    if gn.get("products"):
        prods = [p for p in gn["products"] if p]
        if prods:
            lines.append(f"Products: {', '.join(prods[:15])}")  # cap at 15
    if gn.get("technologies"):
        techs = [t for t in gn["technologies"] if t]
        if techs:
            lines.append(f"Technologies: {', '.join(techs[:15])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer prompt
# ---------------------------------------------------------------------------

_GRAPHRAG_SYSTEM = """\
You answer questions about an AI-startup knowledge graph. You have been
given a set of retrieved documents and structured facts from the graph.

""" + ANSWER_STYLE_GUIDE + """

Additional rules for this retrieval-augmented approach:
- Prefer structured graph facts (funding amounts, investor names, product
  names) over free-text when both are available — they are more reliable.
- If the retrieved documents don't contain enough information to answer
  the question, say so honestly.
- When citing, mention the document title and publisher.
"""


def _build_graphrag_prompt(
    question: str,
    context_blocks: list[str],
) -> str:
    parts = [
        _GRAPHRAG_SYSTEM,
        "RETRIEVED CONTEXT:",
        "\n\n".join(context_blocks),
        f"\nQuestion: {question}",
        "Answer:",
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GraphRAG pipeline
# ---------------------------------------------------------------------------

class GraphRAG:
    """Approach B: vector retrieval + graph expansion + LLM answer."""

    name = "graph_rag"

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        model: str = "llama3.1:8b",
        top_k: int = 5,
    ) -> None:
        self._owns_client = neo4j_client is None
        self.client = neo4j_client or Neo4jClient()
        if self._owns_client:
            self.client.connect()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore(self.client)
        self.model = model
        self.top_k = top_k

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "GraphRAG":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        trace: dict[str, Any] = {}

        # 1. Embed the question
        try:
            q_vec = self.embedder.embed_one(question)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return Answer(
                question=question, text="", approach=self.name,
                latency_ms=elapsed, error=f"Embedding failed: {e}",
            )

        # 2. Vector search for top-K documents
        try:
            retrieved = self.vector_store.query(q_vec, k=self.top_k)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return Answer(
                question=question, text="", approach=self.name,
                latency_ms=elapsed, error=f"Vector search failed: {e}",
            )

        trace["retrieved_docs"] = [
        {
            "doc_id":         r.get("doc_id"),
            "score":          r.get("score"),
            "title":          r.get("title"),
            "url":            r.get("url"),            # ← this is what makes links work
            "publisher":      r.get("publisher"),
            "published_date": r.get("published_date"),
        }
        for r in retrieved
]

        if not retrieved:
            elapsed = (time.perf_counter() - start) * 1000
            return Answer(
                question=question,
                text="I couldn't find any relevant documents in the knowledge graph.",
                approach=self.name, latency_ms=elapsed, trace=trace,
            )

        # 3. Graph expansion + context assembly
        context_blocks: list[str] = []
        all_doc_ids: list[str] = []

        for doc in retrieved:
            doc_id = doc.get("doc_id", "")
            all_doc_ids.append(doc_id)

            # Fetch the graph neighborhood
            try:
                rows = self.client.run_read(
                    _EXPAND_NEIGHBORHOOD, {"doc_id": doc_id}
                )
                gn = rows[0] if rows else {}
            except Exception:
                gn = {}

            # Fetch the document text for the snippet
            try:
                text_rows = self.client.run_read(
                    "MATCH (d:Document {doc_id: $doc_id}) "
                    "RETURN d.title AS title",
                    {"doc_id": doc_id},
                )
                # We don't store cleaned_text in Neo4j — just use the title
                # and the graph neighborhood for context.
                text_snippet = doc.get("title", "")
            except Exception:
                text_snippet = ""

            block = _build_context_block(doc, text_snippet, gn)
            context_blocks.append(block)

        trace["num_context_blocks"] = len(context_blocks)

        # 4. LLM answer generation
        prompt = _build_graphrag_prompt(question, context_blocks)
        trace["prompt_length"] = len(prompt)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            text = response.get("message", {}).get("content", "").strip()
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return Answer(
                question=question, text="", approach=self.name,
                latency_ms=elapsed, trace=trace,
                error=f"LLM call failed: {e}",
            )

        if not text:
            elapsed = (time.perf_counter() - start) * 1000
            return Answer(
                question=question, text="", approach=self.name,
                latency_ms=elapsed, trace=trace,
                error="Empty response from LLM",
            )

        elapsed = (time.perf_counter() - start) * 1000
        return Answer(
            question=question,
            text=text,
            cited_doc_ids=all_doc_ids,
            approach=self.name,
            latency_ms=elapsed,
            trace=trace,
        )
