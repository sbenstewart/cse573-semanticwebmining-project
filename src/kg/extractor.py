"""LLM-based funding-round extractor (Phase 2, Pass 1).

Takes a document from ``corpus.jsonl`` and asks a local Ollama model to
extract any funding-round facts mentioned in the text. Returns a list of
``FundingRound`` dicts (empty list if none found).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

import ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1:8b"
MAX_TEXT_CHARS = 2500  # truncate input text; enough for most news articles

PROMPT_TEMPLATE = """You are a financial data extraction system. Extract funding rounds
mentioned in the article below. Return a JSON object with EXACTLY this shape:

{{
  "rounds": [
    {{
      "company": "<company name, as written>",
      "amount_raw": "<amount as written, e.g. '$400M', '€11M', '$1.2B'>",
      "amount_usd": <integer dollars (e.g. 400000000), or null if not stated>,
      "round_type": "<Seed|Pre-Seed|Series A|Series B|Series C|Series D|Series E|Growth|Debt|Unknown>",
      "valuation_raw": "<valuation as written, or null>",
      "valuation_usd": <integer dollars, or null>,
      "announced_date": "<YYYY-MM-DD or YYYY-MM, or null>",
      "investors": ["<investor name>", ...]
    }}
  ]
}}

Rules:
- If the article does NOT describe a funding round, return {{"rounds": []}}.
- Do NOT invent data. Only extract facts explicitly stated in the text.
- CRITICAL: If the article reports an aggregate total across multiple companies
  (e.g. "Anthropic, Replit, and Higgsfield AI raised over $13 billion combined"),
  do NOT attribute that total to any individual company. Return {{"rounds": []}}
  unless each company's individual raise is stated separately.
- CRITICAL: Only extract a round if the article explicitly attributes the
  specific dollar amount to a specific named company. If two companies are
  mentioned and only one amount is given without clear attribution, return
  {{"rounds": []}}.
- "amount_usd" should be the dollar value of the raise, converted from other
  currencies using a rough 1:1 USD estimate (better an estimate than null).
- "round_type" must be one of the listed values; use "Unknown" if unclear.
- "investors" is a flat list of all investors mentioned (lead + participating).
- Return ONLY the JSON object, no prose, no markdown fences.

ARTICLE TITLE: {title}

ARTICLE TEXT: {text}
"""


@dataclass
class FundingRound:
    """Normalized funding-round fact extracted from one document."""
    company: str
    amount_raw: str | None
    amount_usd: int | None
    round_type: str
    valuation_raw: str | None
    valuation_usd: int | None
    announced_date: str | None
    investors: list[str] = field(default_factory=list)
    source_doc_id: str = ""
    round_id: str = ""

    def compute_round_id(self) -> str:
        """Deterministic ID: sha1(company + amount_usd + year)[:12]. Two
        articles reporting the same raise → same ID → single Neo4j node."""
        norm = (self.company or "").strip().lower()
        amt = str(self.amount_usd or "0")
        year = (self.announced_date or "")[:4] or "0000"
        h = hashlib.sha1(f"{norm}|{amt}|{year}".encode("utf-8")).hexdigest()
        return h[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FundingExtractor:
    """Wraps an Ollama model for funding-round extraction.

    Usage::

        extractor = FundingExtractor()
        rounds = extractor.extract(doc)   # list[FundingRound]
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_chars: int = MAX_TEXT_CHARS):
        self.model = model
        self.max_chars = max_chars

    # --- public API -------------------------------------------------------

    def extract(self, doc: dict) -> list[FundingRound]:
        title = (doc.get("title") or "").strip()
        text = (doc.get("cleaned_text") or doc.get("raw_text") or "")[: self.max_chars]
        if not text:
            return []

        raw_response = self._call_llm(title, text)
        parsed = self._parse_json(raw_response)
        if parsed is None:
            # One retry with a stricter nudge
            raw_response = self._call_llm(
                title, text,
                extra="Your previous response was not valid JSON. Return ONLY the JSON object.",
            )
            parsed = self._parse_json(raw_response)
            if parsed is None:
                logger.warning(
                    f"[Extractor] Failed to parse JSON for doc {doc.get('doc_id')}"
                )
                return []

        return self._to_rounds(parsed, doc.get("doc_id", ""))

    # --- internals --------------------------------------------------------

    def _call_llm(self, title: str, text: str, extra: str = "") -> str:
        prompt = PROMPT_TEMPLATE.format(title=title, text=text)
        if extra:
            prompt = f"{extra}\n\n{prompt}"
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        return resp["message"]["content"]

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage: find the first balanced {...} block
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    def _to_rounds(self, parsed: dict, doc_id: str) -> list[FundingRound]:
        rounds_raw = parsed.get("rounds") or []
        if not isinstance(rounds_raw, list):
            return []

        out: list[FundingRound] = []
        for r in rounds_raw:
            if not isinstance(r, dict):
                continue
            company = (r.get("company") or "").strip()
            if not company:
                continue  # can't link a round to nothing
            amount_usd = self._coerce_int(r.get("amount_usd"))
            if amount_usd is None:
                logger.debug(
                    f"[Extractor] Skipping round for '{company}' in {doc_id}: no amount_usd"
                )
                continue  # without a dollar amount, the round isn't useful

            investors = r.get("investors") or []
            if not isinstance(investors, list):
                investors = []
            investors = [str(i).strip() for i in investors if i]

            round_obj = FundingRound(
                company=company,
                amount_raw=self._coerce_str(r.get("amount_raw")),
                amount_usd=amount_usd,
                round_type=self._coerce_str(r.get("round_type")) or "Unknown",
                valuation_raw=self._coerce_str(r.get("valuation_raw")),
                valuation_usd=self._coerce_int(r.get("valuation_usd")),
                announced_date=self._coerce_str(r.get("announced_date")),
                investors=investors,
                source_doc_id=doc_id,
            )
            round_obj.round_id = round_obj.compute_round_id()
            out.append(round_obj)
        return self._drop_aggregate_duplicates(out, doc_id)

    @staticmethod
    def _drop_aggregate_duplicates(
        rounds: list["FundingRound"], doc_id: str
    ) -> list["FundingRound"]:
        """Catch the 'aggregate attribution' bug deterministically.

        When the LLM sees a roundup article like "Anthropic, Replit, and
        Higgsfield AI raised over $13B combined", it sometimes attributes
        the same dollar+date to all three companies. The signature is:
        same (amount_usd, year) tuple appears for two or more *different*
        companies in the same source document. We drop the entire group —
        we can't tell which company (if any) actually raised the amount.
        """
        if len(rounds) < 2:
            return rounds

        # Group by (amount_usd, year) — case-insensitive on company
        groups: dict[tuple[int, str], list["FundingRound"]] = {}
        for r in rounds:
            year = (r.announced_date or "")[:4] or "0000"
            key = (r.amount_usd or 0, year)
            groups.setdefault(key, []).append(r)

        kept: list["FundingRound"] = []
        for key, group in groups.items():
            distinct_companies = {(r.company or "").strip().lower() for r in group}
            if len(distinct_companies) >= 2:
                logger.info(
                    f"[Extractor] Dropping {len(group)} rounds in {doc_id} as "
                    f"likely aggregate attribution: amount={key[0]} year={key[1]} "
                    f"companies={sorted(distinct_companies)}"
                )
                continue
            kept.extend(group)
        return kept

    @staticmethod
    def _coerce_int(v) -> int | None:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            # LLM sometimes returns "400000000" as str; int() above handles that,
            # but also handles "$400M" poorly. Try stripping.
            if isinstance(v, str):
                cleaned = re.sub(r"[^\d]", "", v)
                if cleaned:
                    try:
                        return int(cleaned)
                    except ValueError:
                        return None
            return None

    @staticmethod
    def _coerce_str(v) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s and s.lower() != "null" else None


# ============================================================================
# PASS 2: Product + Technology extractor
# ============================================================================

PRODUCT_TECH_PROMPT = """You are a product intelligence extraction system. Read the article
below and extract NAMED products and the technologies those products use.

Return a JSON object with EXACTLY this shape:

{{
  "items": [
    {{
      "company": "<company that makes the product>",
      "product": "<specific named product, e.g. 'Claude 3.5 Sonnet', 'Replit Agent', 'GPT-4o'>",
      "description": "<one-sentence description of what it does, or null>",
      "technologies": ["<specific technology, e.g. 'transformer', 'vector database', 'RAG', 'LangChain'>", ...]
    }}
  ]
}}

STRICT RULES — read carefully:
- ONLY extract products with a SPECIFIC PROPER NAME. Examples of valid product names:
  "Claude 3.5", "Replit Agent", "GPT-4o", "AssemblyAI Universal-1", "Cursor IDE".
- DO NOT extract generic descriptions like "AI platform", "the product", "machine
  learning system", "their tool", "an LLM-based service". These are NOT products.
- DO NOT extract company names as products. ("OpenAI" is a company, "GPT-4" is a product.)
- If the article does not name any specific product, return {{"items": []}}.
- "technologies" is a list of specific technologies the product uses (LLM names,
  databases, frameworks). DO NOT include vague terms like "AI", "ML", "deep learning".
- Do NOT invent data. Only extract facts explicitly stated.
- Return ONLY the JSON object, no prose, no markdown fences.

ARTICLE TITLE: {title}

ARTICLE TEXT: {text}
"""


@dataclass
class ProductMention:
    """A named product extracted from one document, with its tech stack."""
    company: str
    product: str
    description: str | None
    technologies: list[str] = field(default_factory=list)
    source_doc_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Phrases that indicate the LLM extracted a generic description rather than
# a real named product. We drop these in a post-filter even if the LLM
# returned them despite the prompt rules.
GENERIC_PRODUCT_BLOCKLIST = {
    "ai platform", "ml platform", "the platform", "the product", "the tool",
    "ai tool", "ai tools", "ai service", "the service", "ai system",
    "machine learning platform", "llm", "an llm", "a model", "the model",
    "ai model", "ai models", "language model", "language models",
    "the company", "their product", "their platform", "the api", "an api",
    "ai", "ml", "deep learning", "generative ai", "agent", "agents",
}


class ProductTechExtractor:
    """LLM-based extractor for named products and the tech they use."""

    def __init__(self, model: str = DEFAULT_MODEL, max_chars: int = MAX_TEXT_CHARS):
        self.model = model
        self.max_chars = max_chars

    def extract(self, doc: dict) -> list[ProductMention]:
        title = (doc.get("title") or "").strip()
        text = (doc.get("cleaned_text") or doc.get("raw_text") or "")[: self.max_chars]
        if not text:
            return []

        raw = self._call_llm(title, text)
        parsed = FundingExtractor._parse_json(raw)
        if parsed is None:
            raw = self._call_llm(
                title, text,
                extra="Your previous response was not valid JSON. Return ONLY the JSON object.",
            )
            parsed = FundingExtractor._parse_json(raw)
            if parsed is None:
                logger.warning(
                    f"[ProductTech] Failed to parse JSON for doc {doc.get('doc_id')}"
                )
                return []

        return self._to_mentions(parsed, doc.get("doc_id", ""))

    def _call_llm(self, title: str, text: str, extra: str = "") -> str:
        prompt = PRODUCT_TECH_PROMPT.format(title=title, text=text)
        if extra:
            prompt = f"{extra}\n\n{prompt}"
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        return resp["message"]["content"]

    def _to_mentions(self, parsed: dict, doc_id: str) -> list[ProductMention]:
        items = parsed.get("items") or []
        if not isinstance(items, list):
            return []

        out: list[ProductMention] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            company = (r.get("company") or "").strip()
            product = (r.get("product") or "").strip()
            if not company or not product:
                continue
            if product.lower() in GENERIC_PRODUCT_BLOCKLIST:
                logger.debug(
                    f"[ProductTech] Dropping generic product '{product}' from {doc_id}"
                )
                continue
            if product.lower() == company.lower():
                logger.debug(
                    f"[ProductTech] Dropping product==company '{product}' from {doc_id}"
                )
                continue

            techs = r.get("technologies") or []
            if not isinstance(techs, list):
                techs = []
            techs = [str(t).strip() for t in techs if t and str(t).strip()]
            techs = [t for t in techs if t.lower() not in GENERIC_PRODUCT_BLOCKLIST]

            out.append(ProductMention(
                company=company,
                product=product,
                description=FundingExtractor._coerce_str(r.get("description")),
                technologies=techs,
                source_doc_id=doc_id,
            ))
        return out
