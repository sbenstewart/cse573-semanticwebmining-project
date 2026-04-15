"""Tests for Phase 3 Step 2: executor, generator, formatter, pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.answer_formatter import (
    AnswerFormatter,
    build_answer_prompt,
    extract_cited_doc_ids,
)
from src.rag.common import Answer
from src.rag.cypher_executor import ExecutionResult, SafeCypherExecutor
from src.rag.cypher_generator import (
    CypherGenerator,
    GenerationResult,
    build_prompt,
)
from src.rag.text_to_cypher import TextToCypherQA


# ---------------------------------------------------------------------------
# SafeCypherExecutor
# ---------------------------------------------------------------------------

class TestSafeCypherExecutor:
    def _mock_client(self, rows=None, exc=None):
        client = MagicMock()
        if exc is not None:
            client.run_read.side_effect = exc
        else:
            client.run_read.return_value = rows or []
        return client

    def test_executes_valid_query(self):
        client = self._mock_client(rows=[{"name": "Replit"}])
        executor = SafeCypherExecutor(client)
        result = executor.execute("MATCH (s:Startup) RETURN s.name LIMIT 5")
        assert result.ok
        assert result.rows == [{"name": "Replit"}]
        assert result.error is None
        assert result.latency_ms >= 0
        client.run_read.assert_called_once()

    def test_rejects_destructive_query_before_hitting_neo4j(self):
        client = self._mock_client(rows=[])
        executor = SafeCypherExecutor(client)
        result = executor.execute("MATCH (n) DETACH DELETE n")
        assert not result.ok
        assert "Unsafe Cypher" in result.error
        # Critical: the destructive query must never have been sent.
        client.run_read.assert_not_called()

    def test_catches_neo4j_exception(self):
        client = self._mock_client(exc=RuntimeError("syntax error near LIMIT"))
        executor = SafeCypherExecutor(client)
        result = executor.execute("MATCH (s:Startup) RETURN s.name")
        assert not result.ok
        assert "RuntimeError" in result.error
        assert "syntax error" in result.error

    def test_truncates_runaway_results(self):
        client = self._mock_client(rows=[{"i": i} for i in range(1000)])
        executor = SafeCypherExecutor(client, row_limit=50)
        result = executor.execute("MATCH (s:Startup) RETURN s.name")
        assert result.ok
        assert len(result.rows) == 50

    def test_execution_result_dataclass(self):
        r = ExecutionResult(
            rows=[{"x": 1}], error=None, latency_ms=5.0,
            cypher="MATCH (n) RETURN n",
        )
        assert r.ok
        r2 = ExecutionResult(
            rows=[], error="boom", latency_ms=0.1, cypher="MATCH ...",
        )
        assert not r2.ok


# ---------------------------------------------------------------------------
# CypherGenerator
# ---------------------------------------------------------------------------

class TestCypherGenerator:
    def _mock_ollama(self, content: str):
        return {"message": {"content": content}}

    def test_builds_prompt_with_schema_and_examples(self):
        prompt = build_prompt("Who invested in Replit?")
        # Must include schema details, examples, and the question
        assert "Startup" in prompt
        assert "HAS_FUNDING_ROUND" in prompt
        assert "normalized_name" in prompt
        assert "Who invested in Replit?" in prompt
        assert "EXAMPLES" in prompt
        # Must forbid destructive ops
        assert "CREATE" in prompt
        assert "DELETE" in prompt

    def test_retry_error_included_in_prompt(self):
        prompt = build_prompt(
            "x", retry_error="SyntaxError: unknown variable 'foo'"
        )
        assert "previous attempt failed" in prompt
        assert "SyntaxError" in prompt

    def test_parse_clean_json(self):
        gen = CypherGenerator()
        with patch("ollama.chat") as m:
            m.return_value = self._mock_ollama(
                '{"cypher": "MATCH (s:Startup) RETURN s.name",'
                ' "rationale": "direct lookup"}'
            )
            result = gen.generate("List all startups")
        assert result.ok
        assert result.cypher == "MATCH (s:Startup) RETURN s.name"
        assert result.rationale == "direct lookup"
        assert result.error is None

    def test_parse_strips_trailing_semicolon(self):
        gen = CypherGenerator()
        with patch("ollama.chat") as m:
            m.return_value = self._mock_ollama(
                '{"cypher": "MATCH (s:Startup) RETURN s.name;",'
                ' "rationale": "x"}'
            )
            result = gen.generate("q")
        assert result.ok
        assert result.cypher == "MATCH (s:Startup) RETURN s.name"

    def test_parse_salvages_embedded_json(self):
        gen = CypherGenerator()
        with patch("ollama.chat") as m:
            m.return_value = self._mock_ollama(
                'Sure, here it is:\n'
                '{"cypher": "MATCH (n) RETURN n", "rationale": "note"}\n'
                'hope this helps!'
            )
            result = gen.generate("q")
        assert result.ok
        assert result.cypher == "MATCH (n) RETURN n"

    def test_parse_empty_response_errors(self):
        gen = CypherGenerator()
        with patch("ollama.chat") as m:
            m.return_value = self._mock_ollama("")
            result = gen.generate("q")
        assert not result.ok
        assert "Empty" in result.error

    def test_parse_malformed_json_errors(self):
        gen = CypherGenerator()
        with patch("ollama.chat") as m:
            m.return_value = self._mock_ollama("not json at all")
            result = gen.generate("q")
        assert not result.ok
        assert "No JSON object" in result.error or "Malformed" in result.error

    def test_parse_missing_cypher_field(self):
        gen = CypherGenerator()
        with patch("ollama.chat") as m:
            m.return_value = self._mock_ollama(
                '{"rationale": "but I forgot the cypher"}'
            )
            result = gen.generate("q")
        assert not result.ok
        assert "missing 'cypher'" in result.error

    def test_ollama_exception_handled(self):
        gen = CypherGenerator()
        with patch("ollama.chat", side_effect=RuntimeError("connection refused")):
            result = gen.generate("q")
        assert not result.ok
        assert "Ollama call failed" in result.error


# ---------------------------------------------------------------------------
# AnswerFormatter
# ---------------------------------------------------------------------------

class TestAnswerFormatter:
    def test_empty_rows_short_circuit(self):
        fmt = AnswerFormatter()
        # Must NOT call ollama when there are no rows.
        with patch("ollama.chat") as m:
            text, err = fmt.format("q", [])
        assert err is None
        assert "couldn't find" in text.lower()
        m.assert_not_called()

    def test_format_calls_llm_with_rows(self):
        fmt = AnswerFormatter()
        with patch("ollama.chat") as m:
            m.return_value = {"message": {"content": "Replit raised $400M."}}
            text, err = fmt.format(
                "How much did Replit raise?",
                [{"startup": "Replit", "amount": "$400M"}],
                cypher="MATCH ...",
            )
        assert err is None
        assert text == "Replit raised $400M."
        m.assert_called_once()

    def test_format_handles_ollama_error(self):
        fmt = AnswerFormatter()
        with patch("ollama.chat", side_effect=RuntimeError("timeout")):
            text, err = fmt.format("q", [{"x": 1}])
        assert text == ""
        assert "Ollama call failed" in err

    def test_build_answer_prompt_contains_context(self):
        prompt = build_answer_prompt(
            "q?", [{"a": 1}, {"a": 2}], cypher="MATCH (n) RETURN n"
        )
        assert "q?" in prompt
        assert '"a": 1' in prompt
        assert "MATCH (n) RETURN n" in prompt
        assert "2 rows" in prompt


class TestExtractCitedDocIds:
    def test_empty(self):
        assert extract_cited_doc_ids([]) == []

    def test_top_level_doc_id(self):
        rows = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "a"}]
        assert extract_cited_doc_ids(rows) == ["a", "b"]

    def test_nested_doc_id(self):
        rows = [{"round": {"amount": 100, "doc_id": "d1"}}]
        assert extract_cited_doc_ids(rows) == ["d1"]

    def test_mixed(self):
        rows = [
            {"doc_id": "top1"},
            {"meta": {"doc_id": "nested1"}},
            {"doc_id": "top1"},  # duplicate
            {"items": [{"doc_id": "deep1"}]},
        ]
        assert extract_cited_doc_ids(rows) == ["top1", "nested1", "deep1"]

    def test_no_doc_ids(self):
        assert extract_cited_doc_ids([{"name": "Replit"}]) == []


# ---------------------------------------------------------------------------
# End-to-end pipeline (both LLM calls mocked, Neo4j mocked)
# ---------------------------------------------------------------------------

class TestTextToCypherQAPipeline:
    def _pipeline(self, *, gen_result, exec_rows, exec_error=None,
                  format_text="A nice answer.", format_error=None):
        """Build a fully-mocked pipeline for one test."""
        gen = MagicMock()
        gen.generate.return_value = gen_result

        client = MagicMock()
        if exec_error:
            client.run_read.side_effect = RuntimeError(exec_error)
        else:
            client.run_read.return_value = exec_rows

        fmt = MagicMock()
        fmt.format.return_value = (format_text, format_error)

        qa = TextToCypherQA(
            neo4j_client=client, generator=gen, formatter=fmt,
        )
        return qa, gen, client, fmt

    def test_happy_path(self):
        qa, gen, client, fmt = self._pipeline(
            gen_result=GenerationResult(
                cypher="MATCH (s:Startup) RETURN s.name",
                rationale="direct lookup",
                error=None,
                raw_response='{"cypher": "...", "rationale": "..."}',
            ),
            exec_rows=[
                {"startup": "Replit", "doc_id": "doc-123"},
                {"startup": "Anthropic", "doc_id": "doc-456"},
            ],
            format_text="Replit and Anthropic are listed.",
        )
        ans = qa.answer("Who has funding rounds?")
        assert isinstance(ans, Answer)
        assert not ans.is_error()
        assert ans.approach == "text_to_cypher"
        assert ans.text == "Replit and Anthropic are listed."
        assert ans.cited_doc_ids == ["doc-123", "doc-456"]
        assert ans.latency_ms >= 0
        assert len(ans.trace["attempts"]) == 1
        assert ans.trace["attempts"][0]["cypher"].startswith("MATCH")

    def test_generation_failure_returns_error_answer(self):
        qa, gen, client, fmt = self._pipeline(
            gen_result=GenerationResult(
                cypher=None, rationale="",
                error="Empty response from LLM",
                raw_response="",
            ),
            exec_rows=[],
        )
        ans = qa.answer("bad question")
        assert ans.is_error()
        assert "Empty response" in ans.error
        # Formatter must NOT be called on error path
        fmt.format.assert_not_called()

    def test_safety_rejection_returns_error_answer(self):
        qa, gen, client, fmt = self._pipeline(
            gen_result=GenerationResult(
                cypher="DELETE FROM Startup",  # trips safety validator
                rationale="malicious",
                error=None,
                raw_response="{}",
            ),
            exec_rows=[],
        )
        # Disable retry so we only see one attempt
        qa.max_retries = 0
        ans = qa.answer("hack the graph")
        assert ans.is_error()
        assert "Unsafe Cypher" in ans.error
        # Neo4j must never have been called
        client.run_read.assert_not_called()

    def test_retry_on_execution_error(self):
        """First Cypher fails, second succeeds. Happy retry path."""
        # Generator returns one bad then one good cypher on successive calls
        gen = MagicMock()
        gen.generate.side_effect = [
            GenerationResult(
                cypher="MATCH (s:Startup WHERE s.bad RETURN s",  # syntax error
                rationale="attempt 1",
                error=None,
                raw_response="{}",
            ),
            GenerationResult(
                cypher="MATCH (s:Startup) RETURN s.name",
                rationale="attempt 2",
                error=None,
                raw_response="{}",
            ),
        ]

        client = MagicMock()
        client.run_read.side_effect = [
            RuntimeError("SyntaxError"),
            [{"startup": "Replit"}],
        ]

        fmt = MagicMock()
        fmt.format.return_value = ("Replit.", None)

        qa = TextToCypherQA(
            neo4j_client=client, generator=gen, formatter=fmt, max_retries=1,
        )
        ans = qa.answer("q")
        assert not ans.is_error()
        assert ans.text == "Replit."
        assert len(ans.trace["attempts"]) == 2
        # Second attempt's retry_error should have been passed in
        call_args_list = gen.generate.call_args_list
        assert call_args_list[0].kwargs.get("retry_error") is None
        assert "SyntaxError" in (call_args_list[1].kwargs.get("retry_error") or "")
