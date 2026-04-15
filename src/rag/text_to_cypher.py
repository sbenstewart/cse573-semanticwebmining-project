"""Approach A: Text-to-Cypher QA pipeline.

Wires together:
  question
    -> CypherGenerator  (LLM produces Cypher)
    -> SafeCypherExecutor (validator + Neo4j)
    -> retry once on error
    -> AnswerFormatter  (LLM produces English)
    -> Answer

Implements the BaseQASystem protocol from common.py so the Phase 4
benchmark can treat it interchangeably with Approach B.
"""
from __future__ import annotations

import logging
import time

from src.kg.neo4j_client import Neo4jClient
from src.rag.answer_formatter import AnswerFormatter, extract_cited_doc_ids
from src.rag.common import Answer
from src.rag.cypher_executor import SafeCypherExecutor
from src.rag.cypher_generator import CypherGenerator

logger = logging.getLogger(__name__)


class TextToCypherQA:
    """Classical baseline: NL question -> Cypher -> rows -> English answer."""

    name = "text_to_cypher"

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        generator: CypherGenerator | None = None,
        formatter: AnswerFormatter | None = None,
        max_retries: int = 1,
    ) -> None:
        # The client is the only piece that talks to a live service in
        # tests; we accept an optional one so tests can inject a mock.
        self._owns_client = neo4j_client is None
        self.client = neo4j_client or Neo4jClient()
        if self._owns_client:
            self.client.connect()
        self.executor = SafeCypherExecutor(self.client)
        self.generator = generator or CypherGenerator()
        self.formatter = formatter or AnswerFormatter()
        self.max_retries = max_retries

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "TextToCypherQA":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        trace: dict = {"attempts": []}

        # Try generation + execution, with up to max_retries retries
        # feeding the previous error back to the LLM.
        last_error: str | None = None
        rows: list[dict] = []
        cypher: str | None = None

        for attempt in range(self.max_retries + 1):
            gen = self.generator.generate(question, retry_error=last_error)
            attempt_trace = {
                "attempt": attempt + 1,
                "rationale": gen.rationale,
                "cypher": gen.cypher,
                "gen_error": gen.error,
            }

            if not gen.ok:
                trace["attempts"].append(attempt_trace)
                last_error = gen.error
                continue

            cypher = gen.cypher
            exec_result = self.executor.execute(cypher)
            attempt_trace["exec_error"] = exec_result.error
            attempt_trace["exec_latency_ms"] = exec_result.latency_ms
            attempt_trace["rows_returned"] = len(exec_result.rows)
            trace["attempts"].append(attempt_trace)

            if exec_result.ok:
                rows = exec_result.rows
                last_error = None
                break
            last_error = exec_result.error

        elapsed = (time.perf_counter() - start) * 1000

        if last_error:
            return Answer(
                question=question,
                text="",
                approach=self.name,
                latency_ms=elapsed,
                trace=trace,
                error=last_error,
            )

        # Format the answer with the LLM.
        text, fmt_error = self.formatter.format(question, rows, cypher=cypher)
        trace["format_error"] = fmt_error
        cited = extract_cited_doc_ids(rows)

        elapsed = (time.perf_counter() - start) * 1000
        return Answer(
            question=question,
            text=text,
            cited_doc_ids=cited,
            approach=self.name,
            latency_ms=elapsed,
            trace=trace,
            error=fmt_error,
        )
