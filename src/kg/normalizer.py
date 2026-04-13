"""Name normalization for KG entities.

The KG uses ``normalized_name`` as the uniqueness key for Startup/Investor/
Product/Technology/Skill/Event nodes. This module defines exactly how that
normalization works, plus a seed alias table for common VC-firm variants
("a16z" → "andreessen horowitz") so they collapse to a single node.
"""
from __future__ import annotations

import re
import unicodedata

# Seed alias table. LHS is what might appear in articles, RHS is the canonical
# form we normalize everything to. Matches are done on the *normalized* LHS
# (lowercased, punctuation-stripped) so "A16Z", "a16z", and "A-16-Z" all hit.
# Extend this list as you notice duplicates in the graph.
INVESTOR_ALIASES: dict[str, str] = {
    "a16z": "Andreessen Horowitz",
    "andreessen horowitz": "Andreessen Horowitz",
    "andreessenhorowitz": "Andreessen Horowitz",
    "ah capital": "Andreessen Horowitz",
    "sequoia": "Sequoia Capital",
    "sequoia capital": "Sequoia Capital",
    "kpcb": "Kleiner Perkins",
    "kleiner perkins": "Kleiner Perkins",
    "kleiner perkins caufield byers": "Kleiner Perkins",
    "accel": "Accel",
    "accel partners": "Accel",
    "benchmark": "Benchmark",
    "benchmark capital": "Benchmark",
    "founders fund": "Founders Fund",
    "khosla": "Khosla Ventures",
    "khosla ventures": "Khosla Ventures",
    "gv": "GV",
    "google ventures": "GV",
    "nea": "New Enterprise Associates",
    "new enterprise associates": "New Enterprise Associates",
    "ycombinator": "Y Combinator",
    "y combinator": "Y Combinator",
    "yc": "Y Combinator",
    "lightspeed": "Lightspeed Venture Partners",
    "lightspeed venture partners": "Lightspeed Venture Partners",
    "index": "Index Ventures",
    "index ventures": "Index Ventures",
    "greylock": "Greylock",
    "greylock partners": "Greylock",
    "first round": "First Round Capital",
    "first round capital": "First Round Capital",
    "tiger global": "Tiger Global",
    "tiger global management": "Tiger Global",
    "softbank": "SoftBank",
    "softbank vision fund": "SoftBank",
    "insight partners": "Insight Partners",
    "coatue": "Coatue",
    "general catalyst": "General Catalyst",
    "bessemer": "Bessemer Venture Partners",
    "bessemer venture partners": "Bessemer Venture Partners",
}


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_LEGAL_SUFFIXES = (
    " inc", " inc.", " llc", " ltd", " limited", " corp", " corporation",
    " co", " co.", " ag", " sa", " plc", " gmbh",
)


def normalize_name(name: str) -> str:
    """Return the canonical form of an entity name used for uniqueness matching.

    - Unicode NFKC normalization (folds fancy quotes, full-width chars)
    - Strip common legal suffixes ("Inc", "LLC", ...)
    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).strip().lower()
    # Strip legal suffix (only if trailing)
    for suf in _LEGAL_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def canonical_investor(name: str) -> tuple[str, str]:
    """Resolve an investor name against the alias table.

    Returns ``(display_name, normalized_name)``. If the input matches an
    alias, both values come from the canonical RHS. Otherwise we keep the
    caller's surface form for display and just normalize for the key.
    """
    norm = normalize_name(name)
    if norm in INVESTOR_ALIASES:
        canonical = INVESTOR_ALIASES[norm]
        return canonical, normalize_name(canonical)
    return name.strip(), norm


def canonical_startup(name: str) -> tuple[str, str]:
    """Return ``(display_name, normalized_name)`` for a startup.

    Uses a small seed alias table for well-known startups that the LLM is
    likely to extract under multiple surface forms ("Scale" / "Scale AI",
    "Assembly AI" / "AssemblyAI"). Extend as needed.
    """
    norm = normalize_name(name)
    if norm in STARTUP_ALIASES:
        canonical = STARTUP_ALIASES[norm]
        return canonical, normalize_name(canonical)
    return name.strip(), norm


# Startup aliases. Keep short and high-confidence — startup names are more
# ambiguous than investor names, so only add entries where we're sure two
# surface forms refer to the same company.
STARTUP_ALIASES: dict[str, str] = {
    "scale": "Scale AI",
    "scale ai": "Scale AI",
    "assembly ai": "AssemblyAI",
    "assemblyai": "AssemblyAI",
    "open ai": "OpenAI",
    "openai": "OpenAI",
    "anthropic ai": "Anthropic",
    "anthropic": "Anthropic",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
}
