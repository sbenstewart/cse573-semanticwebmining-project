"""Read-only Cypher validator.

Approach A (text-to-Cypher) has an LLM generate Cypher queries that run
against the live Phase 2 knowledge graph. Without guards, a prompt-injection
attack — or just a hallucinating model — could generate destructive
statements like ``MATCH (n) DETACH DELETE n`` and wipe the KG.

This module provides ``validate_read_only(cypher)`` which raises
``UnsafeCypherError`` if the query contains any write, schema-modifying,
or procedure-call operation. It is the single choke point that sits
between the LLM and the Neo4j driver.

Design:
- Token-based scan, case-insensitive.
- Word-boundary matching so "CREATED" or "DELETED" in a string literal
  or property name doesn't trigger false positives.
- String literals and comments are stripped before scanning so the
  keywords inside them don't trip the validator.
- CALL is forbidden outright. Read-only APOC procs (e.g. apoc.text.*)
  exist, but allowlisting individual procedures is a support burden we
  don't need right now. If we later want to allow specific read-only
  procs, we extend the allowlist, not the general CALL permission.
"""
from __future__ import annotations

import re


class UnsafeCypherError(ValueError):
    """Raised when a Cypher query contains a forbidden operation."""


# Operations that modify data, schema, or invoke procedures.
# Every one of these is forbidden in an LLM-generated query. Listed
# as regex alternatives in a single compiled pattern for speed.
_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "CALL",
    "LOAD",          # LOAD CSV
    "FOREACH",
    "USING",         # USING PERIODIC COMMIT
    "START",         # legacy Neo4j procedure start
    "GRANT",
    "REVOKE",
    "DENY",
    "SHOW",          # SHOW CONSTRAINTS / SHOW INDEXES expose metadata
    "TERMINATE",
)

_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Strip string literals (single or double quoted) and single-line
# comments so keywords inside them don't trigger the validator.
_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_literals_and_comments(cypher: str) -> str:
    """Remove strings and comments so their contents aren't scanned."""
    cypher = _BLOCK_COMMENT.sub(" ", cypher)
    cypher = _LINE_COMMENT.sub(" ", cypher)
    cypher = _STRING_LITERAL.sub("''", cypher)
    return cypher


def validate_read_only(cypher: str) -> None:
    """Raise UnsafeCypherError if ``cypher`` contains a forbidden op.

    The validator is conservative: false positives (rejecting a safe
    query) are vastly preferable to false negatives (allowing a write).
    If a legitimate read query gets rejected, the user can rephrase it
    or we can extend the allowlist explicitly.
    """
    if not cypher or not cypher.strip():
        raise UnsafeCypherError("Empty query")

    # Must begin with MATCH, RETURN, WITH, UNWIND, or OPTIONAL MATCH.
    # (Other read-only starts exist but this covers 99% of real queries.)
    stripped = _strip_literals_and_comments(cypher).strip()
    first_token_match = re.match(r"\s*([A-Za-z]+)", stripped)
    if not first_token_match:
        raise UnsafeCypherError("Could not identify the first clause")
    first_token = first_token_match.group(1).upper()
    allowed_starts = {"MATCH", "RETURN", "WITH", "UNWIND", "OPTIONAL"}
    if first_token not in allowed_starts:
        raise UnsafeCypherError(
            f"Query must start with one of {sorted(allowed_starts)}, "
            f"got '{first_token}'"
        )

    # No forbidden keyword anywhere in the (literal-stripped) body.
    hit = _FORBIDDEN_PATTERN.search(stripped)
    if hit:
        raise UnsafeCypherError(
            f"Forbidden operation '{hit.group(1).upper()}' is not allowed "
            f"in read-only mode"
        )


def is_read_only(cypher: str) -> bool:
    """Boolean convenience wrapper. Does not raise."""
    try:
        validate_read_only(cypher)
        return True
    except UnsafeCypherError:
        return False
