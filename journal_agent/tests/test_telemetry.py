"""Tests for TelemetryCallbackHandler (#8-hardening).

Verifies that every LLM call — regardless of provider response shape —
produces a structured log record with the expected fields, and that LLM
errors produce an error-level record.
"""

import logging
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from journal_agent.telemetry import TelemetryCallbackHandler, _token_counts


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: _token_counts normalisation
# ═══════════════════════════════════════════════════════════════════════════════


def test_token_counts_openai_style():
    llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}}
    counts = _token_counts(llm_output)
    assert counts == {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}


def test_token_counts_anthropic_style():
    llm_output = {"usage": {"input_tokens": 80, "output_tokens": 30}}
    counts = _token_counts(llm_output)
    assert counts["prompt_tokens"] == 80
    assert counts["completion_tokens"] == 30
    assert counts["total_tokens"] == 110


def test_token_counts_computes_total_when_missing():
    llm_output = {"token_usage": {"prompt_tokens": 50, "completion_tokens": 20}}
    counts = _token_counts(llm_output)
    assert counts["total_tokens"] == 70


def test_token_counts_empty_llm_output():
    counts = _token_counts({})
    assert counts == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: on_llm_end — structured log record
# ═══════════════════════════════════════════════════════════════════════════════


def _make_llm_result(llm_output: dict):
    from langchain_core.outputs import LLMResult
    return LLMResult(generations=[], llm_output=llm_output)


def test_on_llm_end_logs_at_info(caplog):
    handler = TelemetryCallbackHandler()
    result = _make_llm_result({
        "model_name": "gpt-4o-mini",
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    with caplog.at_level(logging.INFO, logger="journal_agent.telemetry"):
        handler.on_llm_end(result, run_id=uuid4())

    assert any("llm call completed" in r.message for r in caplog.records)


def test_on_llm_end_record_has_token_fields(caplog):
    handler = TelemetryCallbackHandler()
    result = _make_llm_result({
        "model_name": "gpt-4o-mini",
        "token_usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    })
    with caplog.at_level(logging.INFO, logger="journal_agent.telemetry"):
        handler.on_llm_end(result, run_id=uuid4())

    record = next(r for r in caplog.records if "llm call completed" in r.message)
    assert record.prompt_tokens == 12
    assert record.completion_tokens == 8
    assert record.total_tokens == 20


def test_on_llm_end_record_has_model_field(caplog):
    handler = TelemetryCallbackHandler()
    result = _make_llm_result({"model_name": "claude-3-5-haiku", "usage": {}})
    with caplog.at_level(logging.INFO, logger="journal_agent.telemetry"):
        handler.on_llm_end(result, run_id=uuid4())

    record = next(r for r in caplog.records if "llm call completed" in r.message)
    assert record.model == "claude-3-5-haiku"


def test_on_llm_end_captures_langgraph_node_from_metadata(caplog):
    handler = TelemetryCallbackHandler()
    result = _make_llm_result({"token_usage": {}})
    with caplog.at_level(logging.INFO, logger="journal_agent.telemetry"):
        handler.on_llm_end(
            result,
            run_id=uuid4(),
            metadata={"langgraph_node": "intent_classifier"},
        )

    record = next(r for r in caplog.records if "llm call completed" in r.message)
    assert record.node == "intent_classifier"


def test_on_llm_end_node_defaults_to_unknown_when_no_metadata(caplog):
    handler = TelemetryCallbackHandler()
    result = _make_llm_result({})
    with caplog.at_level(logging.INFO, logger="journal_agent.telemetry"):
        handler.on_llm_end(result, run_id=uuid4())

    record = next(r for r in caplog.records if "llm call completed" in r.message)
    assert record.node == "unknown"


def test_on_llm_end_anthropic_token_format(caplog):
    handler = TelemetryCallbackHandler()
    result = _make_llm_result({
        "model_name": "claude-3-5-sonnet",
        "usage": {"input_tokens": 200, "output_tokens": 75},
    })
    with caplog.at_level(logging.INFO, logger="journal_agent.telemetry"):
        handler.on_llm_end(result, run_id=uuid4())

    record = next(r for r in caplog.records if "llm call completed" in r.message)
    assert record.prompt_tokens == 200
    assert record.completion_tokens == 75
    assert record.total_tokens == 275


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: on_llm_error — error log record
# ═══════════════════════════════════════════════════════════════════════════════


def test_on_llm_error_logs_at_error_level(caplog):
    handler = TelemetryCallbackHandler()
    with caplog.at_level(logging.ERROR, logger="journal_agent.telemetry"):
        handler.on_llm_error(RuntimeError("rate limit"), run_id=uuid4())

    assert any("llm call failed" in r.message for r in caplog.records)
    record = next(r for r in caplog.records if "llm call failed" in r.message)
    assert record.levelno == logging.ERROR


def test_on_llm_error_record_has_error_field(caplog):
    handler = TelemetryCallbackHandler()
    with caplog.at_level(logging.ERROR, logger="journal_agent.telemetry"):
        handler.on_llm_error(RuntimeError("timeout"), run_id=uuid4())

    record = next(r for r in caplog.records if "llm call failed" in r.message)
    assert "timeout" in record.error


def test_on_llm_error_captures_node_from_metadata(caplog):
    handler = TelemetryCallbackHandler()
    with caplog.at_level(logging.ERROR, logger="journal_agent.telemetry"):
        handler.on_llm_error(
            ValueError("bad response"),
            run_id=uuid4(),
            metadata={"langgraph_node": "thread_classifier"},
        )

    record = next(r for r in caplog.records if "llm call failed" in r.message)
    assert record.node == "thread_classifier"
