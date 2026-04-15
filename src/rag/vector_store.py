"""Neo4j native vector index for document retrieval.

Uses Neo4j 5.11+ vector index to store and query document embeddings
directly on Document nodes. This keeps the "one-database" story: the
vector store IS the graph store, so retrieval and graph expansion
happen in the same system.

Operations:
  - create_index(): one-time setup of the vector index
  - store_embedding(): write an embedding to a Document node
  - store_embeddings_batch(): bulk write via UNWIND for speed
  - query(): cosine similarity search returning top-K documents
  - drop_index(): cleanup for rebuilds
"""
from __future__ import annotations

import logging
from typing import Any

from src.kg.neo4j_client import Neo4jClient
from src.rag.embedder import EMBEDDING_DIM

logger = logging.getLogger(__name__)

INDEX_NAME = "doc_embedding"


class VectorStore:
    """Neo4j-native vector index over Document nodes."""

    def __init__(self, client: Neo4jClient, dim: int = EMBEDDING_DIM) -> None:
        self.client = client
        self.dim = dim

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def create_index(self) -> None:
        """Create the vector index if it doesn't already exist."""
        cypher = (
            f"CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS "
            f"FOR (d:Document) ON (d.embedding) "
            f"OPTIONS {{indexConfig: {{"
            f" `vector.dimensions`: {self.dim},"
            f" `vector.similarity_function`: 'cosine'"
            f"}}}}"
        )
        self.client.run_write(cypher)
        logger.info(
            f"[VectorStore] Index '{INDEX_NAME}' ensured "
            f"(dim={self.dim}, cosine)"
        )

    def drop_index(self) -> None:
        """Drop the vector index (for rebuilds)."""
        self.client.run_write(
            f"DROP INDEX {INDEX_NAME} IF EXISTS"
        )
        logger.info(f"[VectorStore] Index '{INDEX_NAME}' dropped")

    def index_exists(self) -> bool:
        """Check whether the vector index exists."""
        rows = self.client.run_read(
            "SHOW INDEXES WHERE name = $name",
            {"name": INDEX_NAME},
        )
        return len(rows) > 0

    # ------------------------------------------------------------------
    # Write embeddings
    # ------------------------------------------------------------------

    def store_embedding(self, doc_id: str, embedding: list[float]) -> None:
        """Store a single document embedding."""
        self.client.run_write(
            "MATCH (d:Document {doc_id: $doc_id}) "
            "SET d.embedding = $embedding",
            {"doc_id": doc_id, "embedding": embedding},
        )

    def store_embeddings_batch(
        self, items: list[dict[str, Any]], batch_size: int = 50
    ) -> int:
        """Bulk-store embeddings. Each item: {doc_id, embedding}.

        Returns the number of documents updated.
        """
        total = 0
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            result = self.client.run_write(
                "UNWIND $batch AS item "
                "MATCH (d:Document {doc_id: item.doc_id}) "
                "SET d.embedding = item.embedding "
                "RETURN count(d) AS n",
                {"batch": batch},
            )
            if result:
                total += result[0].get("n", 0)
        return total

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self, embedding: list[float], k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the top-K most similar documents.

        Returns list of dicts with keys:
          doc_id, title, publisher, url, published_date, score
        """
        rows = self.client.run_read(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) "
            "YIELD node, score "
            "RETURN node.doc_id AS doc_id, "
            "       node.title AS title, "
            "       node.publisher AS publisher, "
            "       node.url AS url, "
            "       node.published_date AS published_date, "
            "       score",
            {"index": INDEX_NAME, "k": k, "embedding": embedding},
        )
        return rows
