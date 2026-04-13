"""Write extracted funding rounds to Neo4j.

Takes the ``FundingRound`` objects produced by ``FundingExtractor`` and
MERGEs them into the graph so the operation is idempotent — running the
ingester twice on the same corpus produces the same graph, not duplicates.
"""
from __future__ import annotations

import logging
from typing import Iterable

from .extractor import FundingRound
from .neo4j_client import Neo4jClient
from .normalizer import canonical_investor, canonical_startup

logger = logging.getLogger(__name__)


# Document upsert — called once per doc, independent of funding content
MERGE_DOCUMENT = """
MERGE (d:Document {doc_id: $doc_id})
  ON CREATE SET d.title = $title,
                d.publisher = $publisher,
                d.url = $url,
                d.published_date = $published_date
  ON MATCH  SET d.title = coalesce(d.title, $title),
                d.publisher = coalesce(d.publisher, $publisher),
                d.url = coalesce(d.url, $url),
                d.published_date = coalesce(d.published_date, $published_date)
RETURN d.doc_id AS doc_id
"""

# One big MERGE per round: startup, round, investors, document, plus edges
MERGE_FUNDING = """
MERGE (s:Startup {normalized_name: $startup_norm})
  ON CREATE SET s.name = $startup_name, s.first_seen = $doc_date
  ON MATCH  SET s.name = coalesce(s.name, $startup_name),
                s.last_seen = $doc_date

MERGE (r:FundingRound {round_id: $round_id})
  ON CREATE SET r.round_type = $round_type,
                r.amount_usd = $amount_usd,
                r.amount_raw = $amount_raw,
                r.valuation_usd = $valuation_usd,
                r.valuation_raw = $valuation_raw,
                r.announced_date = $announced_date
  ON MATCH  SET r.amount_usd = coalesce(r.amount_usd, $amount_usd),
                r.valuation_usd = coalesce(r.valuation_usd, $valuation_usd),
                r.announced_date = coalesce(r.announced_date, $announced_date)

MERGE (s)-[:HAS_FUNDING_ROUND]->(r)

WITH s, r
MATCH (d:Document {doc_id: $doc_id})
MERGE (r)-[:SOURCED_FROM]->(d)
MERGE (d)-[:MENTIONS]->(s)

WITH r
UNWIND $investors AS inv
  MERGE (i:Investor {normalized_name: inv.normalized_name})
    ON CREATE SET i.name = inv.display_name
  MERGE (r)-[:INVESTED_BY]->(i)
RETURN r.round_id AS round_id
"""


class FundingIngester:
    """Writes FundingRound objects to Neo4j. Caller supplies the Neo4jClient."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def upsert_document(self, doc: dict) -> None:
        self.client.run_write(
            MERGE_DOCUMENT,
            {
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "publisher": doc.get("publisher", ""),
                "url": doc.get("source_url", ""),
                "published_date": doc.get("published_date"),
            },
        )

    def ingest_rounds(self, rounds: Iterable[FundingRound], doc: dict) -> int:
        n = 0
        doc_date = doc.get("published_date")
        for round_obj in rounds:
            startup_display, startup_norm = canonical_startup(round_obj.company)
            if not startup_norm:
                continue

            investor_rows = []
            seen_norms: set[str] = set()
            for raw_name in round_obj.investors:
                display, norm = canonical_investor(raw_name)
                if not norm or norm in seen_norms:
                    continue
                seen_norms.add(norm)
                investor_rows.append({"display_name": display, "normalized_name": norm})

            self.client.run_write(
                MERGE_FUNDING,
                {
                    "startup_norm": startup_norm,
                    "startup_name": startup_display,
                    "round_id": round_obj.round_id,
                    "round_type": round_obj.round_type,
                    "amount_usd": round_obj.amount_usd,
                    "amount_raw": round_obj.amount_raw,
                    "valuation_usd": round_obj.valuation_usd,
                    "valuation_raw": round_obj.valuation_raw,
                    "announced_date": round_obj.announced_date,
                    "doc_id": doc.get("doc_id", ""),
                    "doc_date": doc_date,
                    "investors": investor_rows,
                },
            )
            n += 1
        return n


# ============================================================================
# PASS 2: Product + Technology ingester
# ============================================================================

MERGE_PRODUCT_TECH = """
MERGE (s:Startup {normalized_name: $startup_norm})
  ON CREATE SET s.name = $startup_name, s.first_seen = $doc_date
  ON MATCH  SET s.name = coalesce(s.name, $startup_name),
                s.last_seen = $doc_date

MERGE (p:Product {normalized_name: $product_norm})
  ON CREATE SET p.name = $product_name, p.description = $description
  ON MATCH  SET p.description = coalesce(p.description, $description)

MERGE (s)-[:ANNOUNCED]->(p)

WITH s, p
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:MENTIONS]->(s)
MERGE (d)-[:MENTIONS]->(p)

WITH p
UNWIND $technologies AS tech
  MERGE (t:Technology {normalized_name: tech.normalized_name})
    ON CREATE SET t.name = tech.display_name
  MERGE (p)-[:USES_TECH]->(t)
RETURN p.normalized_name AS product_norm
"""


class ProductTechIngester:
    """Writes ProductMention objects to Neo4j."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def ingest_mentions(self, mentions, doc: dict) -> int:
        from .extractor import ProductMention  # avoid circular import at module load
        from .normalizer import normalize_name

        n = 0
        doc_date = doc.get("published_date")
        for m in mentions:
            startup_display, startup_norm = canonical_startup(m.company)
            product_norm = normalize_name(m.product)
            if not startup_norm or not product_norm:
                continue

            tech_rows = []
            seen = set()
            for raw in m.technologies:
                norm = normalize_name(raw)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                tech_rows.append({"display_name": raw.strip(), "normalized_name": norm})

            self.client.run_write(
                MERGE_PRODUCT_TECH,
                {
                    "startup_norm": startup_norm,
                    "startup_name": startup_display,
                    "product_norm": product_norm,
                    "product_name": m.product.strip(),
                    "description": m.description,
                    "doc_id": doc.get("doc_id", ""),
                    "doc_date": doc_date,
                    "technologies": tech_rows,
                },
            )
            n += 1
        return n
