"""Knowledge graph schema: uniqueness constraints and indexes.

Mirrors the schema diagram in the project docs:

    Nodes:     Startup, Investor, FundingRound, Product, Technology,
               Skill, Event, Document
    Rels:      HAS_FUNDING_ROUND, INVESTED_BY, ANNOUNCED, USES_TECH,
               HIRING_FOR, PARTNERED_WITH, MENTIONS

Running ``apply_schema`` is idempotent: Neo4j's ``IF NOT EXISTS`` clause
makes repeated calls cheap.
"""
from __future__ import annotations

import logging

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# Uniqueness constraints — one per node type that has a natural key.
# For Startup/Investor/Product/Technology/Skill the key is normalized_name
# (lowercased, punctuation-stripped) so "a16z" and "A16Z" collapse.
CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT startup_name IF NOT EXISTS "
    "FOR (s:Startup) REQUIRE s.normalized_name IS UNIQUE",

    "CREATE CONSTRAINT investor_name IF NOT EXISTS "
    "FOR (i:Investor) REQUIRE i.normalized_name IS UNIQUE",

    "CREATE CONSTRAINT product_name IF NOT EXISTS "
    "FOR (p:Product) REQUIRE p.normalized_name IS UNIQUE",

    "CREATE CONSTRAINT technology_name IF NOT EXISTS "
    "FOR (t:Technology) REQUIRE t.normalized_name IS UNIQUE",

    "CREATE CONSTRAINT skill_name IF NOT EXISTS "
    "FOR (sk:Skill) REQUIRE sk.normalized_name IS UNIQUE",

    "CREATE CONSTRAINT event_name IF NOT EXISTS "
    "FOR (e:Event) REQUIRE e.normalized_name IS UNIQUE",

    "CREATE CONSTRAINT funding_round_id IF NOT EXISTS "
    "FOR (r:FundingRound) REQUIRE r.round_id IS UNIQUE",

    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
]

# Property indexes — not unique, just speed up common lookups.
INDEXES: list[str] = [
    "CREATE INDEX startup_display_name IF NOT EXISTS "
    "FOR (s:Startup) ON (s.name)",

    "CREATE INDEX funding_round_date IF NOT EXISTS "
    "FOR (r:FundingRound) ON (r.announced_date)",

    "CREATE INDEX document_published IF NOT EXISTS "
    "FOR (d:Document) ON (d.published_date)",
]


def apply_schema(client: Neo4jClient) -> dict[str, int]:
    """Create all constraints and indexes. Returns counts of each kind applied."""
    logger.info("[Schema] Applying constraints and indexes")
    for cypher in CONSTRAINTS:
        client.run_write(cypher)
    for cypher in INDEXES:
        client.run_write(cypher)
    logger.info(
        f"[Schema] Applied {len(CONSTRAINTS)} constraints and {len(INDEXES)} indexes"
    )
    return {"constraints": len(CONSTRAINTS), "indexes": len(INDEXES)}


def describe_schema(client: Neo4jClient) -> dict:
    """Return a summary of the current schema state (constraints + indexes)."""
    constraints = client.run_read("SHOW CONSTRAINTS")
    indexes = client.run_read("SHOW INDEXES")
    return {"constraints": constraints, "indexes": indexes}
