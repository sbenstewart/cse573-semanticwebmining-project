"""Safe read-only Cypher executor.

Wraps ``Neo4jClient.run_read`` with the ``cypher_safety`` validator so
any Cypher that reaches Neo4j has been checked first. This is the single
choke point between LLM-generated queries and the live graph.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.kg.neo4j_client import Neo4jClient
from src.rag.cypher_safety import UnsafeCypherError, validate_read_only

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Outcome of a Cypher execution attempt.

    ``rows`` is populated on success. ``error`` is populated on failure.
    Exactly one of the two is set in a given result.
    """
    rows: list[dict[str, Any]]
    error: str | None
    latency_ms: float
    cypher: str

    @property
    def ok(self) -> bool:
        return self.error is None


class SafeCypherExecutor:
    """Validate-then-execute wrapper around ``Neo4jClient``.

    Usage::

        with Neo4jClient() as client:
            executor = SafeCypherExecutor(client)
            result = executor.execute("MATCH (s:Startup) RETURN s.name LIMIT 5")
            if result.ok:
                for row in result.rows:
                    print(row)
    """

    def __init__(self, client: Neo4jClient, row_limit: int = 500) -> None:
        self.client = client
        self.row_limit = row_limit

    def execute(self, cypher: str) -> ExecutionResult:
        start = time.perf_counter()

        # Safety check first. If this fails the query never reaches Neo4j.
        try:
            validate_read_only(cypher)
        except UnsafeCypherError as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"[Executor] Safety check rejected query: {e}")
            return ExecutionResult(
                rows=[], error=f"Unsafe Cypher: {e}",
                latency_ms=elapsed, cypher=cypher,
            )

        # Execute. Any Neo4j errors become ExecutionResult.error so the
        # caller (cypher_generator retry loop) can react without a try/except.
        try:
            rows = self.client.run_read(cypher)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"[Executor] Neo4j raised {type(e).__name__}: {e}")
            return ExecutionResult(
                rows=[], error=f"{type(e).__name__}: {e}",
                latency_ms=elapsed, cypher=cypher,
            )

        # Truncate runaway results to avoid blowing up downstream prompts.
        if len(rows) > self.row_limit:
            logger.info(
                f"[Executor] Truncating {len(rows)} rows to {self.row_limit}"
            )
            rows = rows[: self.row_limit]

        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            rows=rows, error=None, latency_ms=elapsed, cypher=cypher,
        )
