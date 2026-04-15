"""Shared types, interfaces, and prompt fragments for Phase 3.

Both Approach A (text-to-Cypher) and Approach B (GraphRAG) return
``Answer`` objects and implement the ``BaseQASystem`` protocol, so the
Phase 4 benchmark harness can treat them interchangeably.

The ``SCHEMA_PROMPT`` string is used by both approaches — A injects it
into the Cypher-generation prompt so the LLM knows the graph structure;
B injects it into the answer-formatting prompt so the LLM knows what
the expanded subgraphs represent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Answer container
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    """Normalized answer produced by any QA system.

    Fields are chosen so the Phase 4 benchmark can score each system on
    textual quality, citation accuracy, and wall-clock latency using a
    single harness regardless of which approach produced the answer.
    """
    question: str
    text: str
    cited_doc_ids: list[str] = field(default_factory=list)
    approach: str = "unknown"          # "text_to_cypher" | "graph_rag"
    latency_ms: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None           # populated if the answer failed

    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "text": self.text,
            "cited_doc_ids": list(self.cited_doc_ids),
            "approach": self.approach,
            "latency_ms": self.latency_ms,
            "trace": self.trace,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Protocol both approaches must implement
# ---------------------------------------------------------------------------

class BaseQASystem(Protocol):
    """Interface that all QA approaches must implement.

    The ``name`` attribute identifies the approach in benchmark output;
    ``answer`` is the single entry point the benchmark harness calls.
    """

    name: str

    def answer(self, question: str) -> Answer: ...


# ---------------------------------------------------------------------------
# Schema prompt — used by both approaches
# ---------------------------------------------------------------------------

SCHEMA_PROMPT = """\
The knowledge graph has the following schema. Use ONLY these node labels,
relationship types, and property names. Do not invent new ones.

NODE LABELS and their key properties:

  (:Startup {normalized_name, name, first_seen, last_seen})
      Startups / companies mentioned in articles or job postings.

  (:Investor {normalized_name, name})
      Venture capital firms and angel investors.

  (:FundingRound {round_id, round_type, amount_usd, amount_raw,
                  valuation_usd, valuation_raw, announced_date})
      A single funding round. round_type is one of:
      Seed, Pre-Seed, Series A, Series B, Series C, Series D, Series E,
      Series F, Growth, Debt, Unknown.
      amount_usd is an integer in dollars (e.g. 400000000 for $400M).

  (:Product {normalized_name, name, description})
      Named products/services made or used by startups
      (e.g. "Claude 3.5 Sonnet", "Replit Agent", "GPT-4o").

  (:Technology {normalized_name, name})
      Technologies used by products (e.g. "RAG", "LangChain",
      "vector database", "transformer", "RLHF").

  (:Document {doc_id, title, publisher, url, published_date})
      Source documents. Every extracted fact is linked back to at
      least one Document.

RELATIONSHIP TYPES:

  (Startup)-[:HAS_FUNDING_ROUND]->(FundingRound)
  (FundingRound)-[:INVESTED_BY]->(Investor)
  (FundingRound)-[:SOURCED_FROM]->(Document)
  (Startup)-[:ANNOUNCED]->(Product)
  (Product)-[:USES_TECH]->(Technology)
  (Document)-[:MENTIONS]->(Startup)
  (Document)-[:MENTIONS]->(Product)

IMPORTANT CONVENTIONS:
- Match Startup / Investor / Product / Technology nodes by their
  `normalized_name` property, which is lowercased with punctuation
  removed (e.g. "replit", "andreessen horowitz", "claude 3 5 sonnet").
- When returning data for the user, project the human-readable `.name`
  property, not `.normalized_name`.
- To include provenance in an answer, also return `.doc_id` and
  `.title` from the source Document(s) via SOURCED_FROM or MENTIONS.
"""


# Used by answer_formatter to give the LLM a concrete example of
# what a good cited answer looks like.
ANSWER_STYLE_GUIDE = """\
Write a concise English answer to the user's question using ONLY the
facts present in the query result rows below. You MUST follow these rules:

- If the result rows are empty, say "I couldn't find that in the knowledge graph."
- Every name, number, date, or fact in your answer MUST appear verbatim in
  one of the result rows. If you cannot find it in the rows, do not say it.
- Do NOT add background knowledge from training data. The user only wants
  what the graph contains, not what you happen to know about these companies.
- Do NOT speculate, summarize beyond the rows, or fill in plausible-sounding
  details. If a row lacks a date or amount, omit it.
- If the rows look like they answer a different question than the user
  asked, say so honestly: "The query returned X, but this may not directly
  answer your question."
- Use the human-readable display names from the rows, not normalized forms.
- Keep the answer under 150 words unless the user asked for a list.
- Never reproduce more than a short phrase from any source document.
"""
