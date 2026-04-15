"""LLM-based answer formatter.

Takes the question, Cypher query results, and any source-document
metadata, and produces a natural-language answer using the
ANSWER_STYLE_GUIDE. Also extracts `cited_doc_ids` from the results
for the benchmark harness.
"""
from __future__ import annotations

import json
import logging

import ollama

from src.rag.common import ANSWER_STYLE_GUIDE

logger = logging.getLogger(__name__)


_FORMATTER_INSTRUCTIONS = """\
You answer questions about an AI-startup knowledge graph using ONLY
the facts provided in the query result below. The query was generated
by another system and then executed against a Neo4j knowledge graph.

""" + ANSWER_STYLE_GUIDE


def _serialize_rows(rows: list[dict]) -> str:
    """Render result rows as compact JSON for the LLM."""
    if not rows:
        return "[] (no rows returned)"
    try:
        return json.dumps(rows, indent=2, default=str)
    except Exception:
        return str(rows)


def build_answer_prompt(
    question: str,
    rows: list[dict],
    cypher: str | None = None,
) -> str:
    parts = [
        _FORMATTER_INSTRUCTIONS,
        f"Question: {question}",
    ]
    if cypher:
        parts.append(f"Cypher executed:\n{cypher}")
    parts.append(f"Query result ({len(rows)} rows):\n{_serialize_rows(rows)}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def extract_cited_doc_ids(rows: list[dict]) -> list[str]:
    """Pull any doc_id fields out of the result rows for citations.

    The Cypher generator is prompted to include document provenance in
    its RETURN clause when relevant. This helper scans rows for any
    'doc_id' field (top-level or nested) and returns a deduped list in
    first-seen order.
    """
    seen: list[str] = []
    seen_set: set[str] = set()

    def visit(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "doc_id" and isinstance(v, str) and v not in seen_set:
                    seen.append(v)
                    seen_set.add(v)
                else:
                    visit(v)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(rows)
    return seen


class AnswerFormatter:
    """LLM-based result -> English formatter."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def format(
        self,
        question: str,
        rows: list[dict],
        cypher: str | None = None,
    ) -> tuple[str, str | None]:
        """Return (answer_text, error). answer_text is '' if error is set."""
        if not rows:
            # Cheap path: no LLM call needed for empty results.
            return (
                "I couldn't find that information in the knowledge graph.",
                None,
            )

        prompt = build_answer_prompt(question, rows, cypher=cypher)
        logger.debug(f"[Formatter] formatting {len(rows)} rows for: {question!r}")

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
        except Exception as e:
            return "", f"Ollama call failed: {type(e).__name__}: {e}"

        text = response.get("message", {}).get("content", "").strip()
        if not text:
            return "", "Empty response from formatter LLM"

        return text, None
