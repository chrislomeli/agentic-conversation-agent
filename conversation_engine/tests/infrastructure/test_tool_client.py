"""
Tests for the tool_client infrastructure.

Covers:
- ToolSpec creation and schema export
- ToolRegistry registration, lookup, duplicate prevention, catalog
- ToolResultEnvelope and metadata
- LocalToolClient: happy path, input validation error, execution error, output validation error
"""

import pytest
from pydantic import BaseModel

from conversation_engine.infrastructure.tool_client import (
    LocalToolClient,
    ToolRegistry,
    ToolResultEnvelope,
    ToolResultMeta,
    ToolSpec,
)

# ── Test models ─────────────────────────────────────────────────────


class AddInput(BaseModel):
    a: int
    b: int


class AddOutput(BaseModel):
    result: int


def add_handler(inp: AddInput) -> AddOutput:
    return AddOutput(result=inp.a + inp.b)


def failing_handler(inp: AddInput) -> AddOutput:
    raise RuntimeError("boom")


def bad_output_handler(inp: AddInput) -> dict:
    return {"wrong_field": 999}


ADD_SPEC = ToolSpec(
    name="add",
    description="Add two numbers",
    input_model=AddInput,
    output_model=AddOutput,
    handler=add_handler,
)


# ── ToolSpec ────────────────────────────────────────────────────────


class TestToolSpec:
    def test_creation(self):
        assert ADD_SPEC.name == "add"
        assert ADD_SPEC.description == "Add two numbers"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            ADD_SPEC.name = "changed"

    def test_input_schema(self):
        schema = ADD_SPEC.input_schema()
        assert "properties" in schema
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]

    def test_output_schema(self):
        schema = ADD_SPEC.output_schema()
        assert "properties" in schema
        assert "result" in schema["properties"]


# ── ToolRegistry ────────────────────────────────────────────────────


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(ADD_SPEC)
        assert reg.get("add") is ADD_SPEC

    def test_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(ADD_SPEC)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ADD_SPEC)

    def test_get_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(ADD_SPEC)
        assert reg.list_tools() == ["add"]

    def test_catalog(self):
        reg = ToolRegistry()
        reg.register(ADD_SPEC)
        cat = reg.catalog()
        assert len(cat) == 1
        assert cat[0]["name"] == "add"
        assert "inputSchema" in cat[0]
        assert "outputSchema" in cat[0]


# ── ToolResultEnvelope ──────────────────────────────────────────────


class TestToolResultEnvelope:
    def test_meta_hash_deterministic(self):
        h1 = ToolResultMeta.hash_args({"a": 1, "b": 2})
        h2 = ToolResultMeta.hash_args({"b": 2, "a": 1})
        assert h1 == h2

    def test_envelope_payload_property(self):
        env = ToolResultEnvelope(
            meta=ToolResultMeta(tool_name="t", input_args={}),
            structured={"x": 42},
        )
        assert env.payload == {"x": 42}

    def test_envelope_payload_none(self):
        env = ToolResultEnvelope(
            meta=ToolResultMeta(tool_name="t", input_args={}),
        )
        assert env.payload == {}

    def test_model_dump_flat_injects_meta(self):
        env = ToolResultEnvelope(
            meta=ToolResultMeta(tool_name="t", input_args={"a": 1}),
            structured={"result": 99},
        )
        flat = env.model_dump_flat()
        assert flat["result"] == 99
        assert "_meta" in flat
        assert flat["_meta"]["tool_name"] == "t"


# ── LocalToolClient ────────────────────────────────────────────────


class TestLocalToolClient:
    def _make_client(self, handler=add_handler) -> LocalToolClient:
        spec = ToolSpec(
            name="add",
            description="Add two numbers",
            input_model=AddInput,
            output_model=AddOutput,
            handler=handler,
        )
        reg = ToolRegistry()
        reg.register(spec)
        return LocalToolClient(reg)

    def test_happy_path(self):
        client = self._make_client()
        env = client.call("add", {"a": 3, "b": 4})

        assert not env.is_error
        assert env.meta.success
        assert env.structured["result"] == 7
        assert env.meta.tool_name == "add"
        assert env.meta.duration_ms >= 0

    def test_input_validation_error(self):
        client = self._make_client()
        env = client.call("add", {"a": "not_an_int", "b": 4})

        assert env.is_error
        assert not env.meta.success
        assert env.meta.error == "input_validation_error"

    def test_execution_error(self):
        client = self._make_client(handler=failing_handler)
        env = client.call("add", {"a": 1, "b": 2})

        assert env.is_error
        assert not env.meta.success
        assert env.meta.error == "execution_error"
        assert "boom" in env.structured["details"]

    def test_output_validation_error(self):
        client = self._make_client(handler=bad_output_handler)
        env = client.call("add", {"a": 1, "b": 2})

        assert env.is_error
        assert not env.meta.success
        assert env.meta.error == "output_validation_error"

    def test_tool_not_found(self):
        client = self._make_client()
        with pytest.raises(KeyError, match="not found"):
            client.call("nonexistent", {})

    def test_list_tools(self):
        client = self._make_client()
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "add"
