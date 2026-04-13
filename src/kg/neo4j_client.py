"""Thin wrapper around the neo4j-python-driver.

Provides a context-managed client with retry-on-transient-failure, and a
``run_write`` / ``run_read`` split so callers don't have to think about
sessions and transactions.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterable

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, TransientError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Lightweight Neo4j client. Use as a context manager::

        with Neo4jClient() as client:
            client.run_write("CREATE (n:Test {id: $id})", {"id": 1})
            rows = client.run_read("MATCH (n:Test) RETURN n.id AS id")
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = "neo4j",
        max_retries: int = 3,
    ) -> None:
        self.uri = uri
        self.user = user
        self.database = database
        self.max_retries = max_retries
        self._driver: Driver | None = None
        self._password = password

    # --- lifecycle --------------------------------------------------------

    def connect(self) -> None:
        if self._driver is not None:
            return
        logger.info(f"[Neo4j] Connecting to {self.uri} as {self.user}")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self._password))
        self._driver.verify_connectivity()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- query helpers ----------------------------------------------------

    @contextmanager
    def session(self) -> Iterable[Session]:
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        with self._driver.session(database=self.database) as s:
            yield s

    def run_write(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        return self._run_with_retry(cypher, params or {}, write=True)

    def run_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        return self._run_with_retry(cypher, params or {}, write=False)

    def _run_with_retry(self, cypher: str, params: dict, write: bool) -> list[dict]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self.session() as s:
                    fn = s.execute_write if write else s.execute_read
                    return fn(lambda tx: [dict(r) for r in tx.run(cypher, **params)])
            except (ServiceUnavailable, TransientError) as e:
                last_exc = e
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    f"[Neo4j] Transient failure (attempt {attempt}/{self.max_retries}): "
                    f"{type(e).__name__}. Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    # --- convenience ------------------------------------------------------

    def node_count(self) -> int:
        rows = self.run_read("MATCH (n) RETURN count(n) AS c")
        return rows[0]["c"] if rows else 0

    def relationship_count(self) -> int:
        rows = self.run_read("MATCH ()-[r]->() RETURN count(r) AS c")
        return rows[0]["c"] if rows else 0

    def wipe(self) -> None:
        """DANGEROUS: delete every node and relationship in the database."""
        logger.warning("[Neo4j] Wiping all nodes and relationships")
        self.run_write("MATCH (n) DETACH DELETE n")
