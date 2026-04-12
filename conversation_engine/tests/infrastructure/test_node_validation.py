"""
Tests for node_validation infrastructure.

Covers:
- NodeResult creation (success and failure)
- validated_node decorator: happy path, validation failure
- handle_error node: reads NodeResult and sets status
"""

import pytest
from typing import Optional
from pydantic import BaseModel, Field

from conversation_engine.infrastructure.node_validation import (
    NodeError,
    NodeResult,
    validated_node,
    handle_error,
)


# ── NodeResult ──────────────────────────────────────────────────────


class TestNodeResult:
    def test_success(self):
        r = NodeResult.success(data={"x": 42})
        assert r.ok is True
        assert r.data == {"x": 42}
        assert r.error is None

    def test_success_no_data(self):
        r = NodeResult.success()
        assert r.ok is True
        assert r.data is None

    def test_failure(self):
        r = NodeResult.failure("BAD_INPUT", "something broke", {"field": "a"})
        assert r.ok is False
        assert r.error.code == "BAD_INPUT"
        assert r.error.message == "something broke"
        assert r.error.details == {"field": "a"}

    def test_failure_default_details(self):
        r = NodeResult.failure("ERR", "msg")
        assert r.error.details == {}


# ── validated_node decorator ────────────────────────────────────────


class MyInput(BaseModel):
    model_config = {"extra": "ignore"}
    name: str
    count: int = 0


class TestValidatedNode:
    def test_happy_path(self):
        @validated_node(MyInput)
        def my_node(inp: MyInput, state: dict) -> dict:
            return {"greeting": f"Hello {inp.name}", "node_result": NodeResult.success()}

        result = my_node({"name": "Alice", "count": 5, "extra_key": "ignored"})
        assert result["greeting"] == "Hello Alice"

    def test_validation_failure(self):
        @validated_node(MyInput)
        def my_node(inp: MyInput, state: dict) -> dict:
            return {"greeting": f"Hello {inp.name}", "node_result": NodeResult.success()}

        # missing required 'name' field
        result = my_node({"count": 5})
        assert "node_result" in result
        assert result["node_result"].ok is False
        assert result["node_result"].error.code == "INVALID_INPUT"

    def test_extra_keys_ignored(self):
        @validated_node(MyInput)
        def my_node(inp: MyInput, state: dict) -> dict:
            return {"value": inp.count, "node_result": NodeResult.success()}

        result = my_node({"name": "Bob", "count": 3, "random_field": True})
        assert result["value"] == 3

    def test_preserves_function_name(self):
        @validated_node(MyInput)
        def my_special_node(inp: MyInput, state: dict) -> dict:
            return {}

        assert my_special_node.__name__ == "my_special_node"


# ── handle_error ────────────────────────────────────────────────────


class TestHandleError:
    def test_sets_error_status(self):
        state = {
            "node_result": NodeResult.failure("TEST_ERR", "test error"),
            "status": "running",
        }
        result = handle_error(state)
        assert result["status"] == "error"

    def test_handles_missing_node_result(self):
        state = {"status": "running"}
        result = handle_error(state)
        assert result["status"] == "error"

    def test_handles_success_node_result(self):
        state = {
            "node_result": NodeResult.success(),
            "status": "running",
        }
        result = handle_error(state)
        assert result["status"] == "error"
