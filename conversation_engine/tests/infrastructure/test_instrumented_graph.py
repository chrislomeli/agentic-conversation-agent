"""
Tests for InstrumentedGraph + Interceptors.

Covers:
- InstrumentedGraph wraps nodes with interceptor hooks
- LoggingInterceptor logs before/after/error
- MetricsInterceptor collects call counts, durations, errors
- Middleware transforms node results
- Broken interceptor does not crash the graph
- Broken middleware DOES propagate
- InstrumentedGraph works as drop-in for StateGraph
"""

import pytest
from typing import Any, TypedDict
from langgraph.graph import START, END

from conversation_engine.infrastructure.instrumented_graph import (
    InstrumentedGraph,
    Interceptor,
    Middleware,
)
from conversation_engine.infrastructure.interceptors import (
    LoggingInterceptor,
    MetricsInterceptor,
)


# ── Test state ──────────────────────────────────────────────────────


class SimpleState(TypedDict):
    value: int
    log: list


def increment(state: SimpleState) -> dict:
    return {"value": state["value"] + 1}


def failing_node(state: SimpleState) -> dict:
    raise RuntimeError("node exploded")


# ── Recording interceptor ──────────────────────────────────────────


class RecordingInterceptor(Interceptor):
    """Records all hook calls for assertions."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def before(self, node_name: str, state: Any) -> None:
        self.calls.append(("before", node_name))

    def after(self, node_name: str, state: Any, result: Any) -> None:
        self.calls.append(("after", node_name))

    def on_error(self, node_name: str, state: Any, error: Exception) -> None:
        self.calls.append(("on_error", node_name))


class BrokenInterceptor(Interceptor):
    """Always raises — tests that broken interceptors don't crash the graph."""

    def before(self, node_name: str, state: Any) -> None:
        raise ValueError("broken before")

    def after(self, node_name: str, state: Any, result: Any) -> None:
        raise ValueError("broken after")

    def on_error(self, node_name: str, state: Any, error: Exception) -> None:
        raise ValueError("broken on_error")


# ── Recording middleware ───────────────────────────────────────────


class AddTagMiddleware(Middleware):
    """Adds a tag to every result dict."""

    def transform(self, node_name: str, state: Any, result: Any) -> Any:
        if isinstance(result, dict):
            result["log"] = state.get("log", []) + [f"tagged:{node_name}"]
        return result


class BrokenMiddleware(Middleware):
    """Always raises — tests that middleware errors propagate."""

    def transform(self, node_name: str, state: Any, result: Any) -> Any:
        raise RuntimeError("middleware exploded")


# ── Tests ───────────────────────────────────────────────────────────


class TestInstrumentedGraph:
    def test_basic_graph_without_interceptors(self):
        """InstrumentedGraph works as drop-in for StateGraph."""
        g = InstrumentedGraph(SimpleState)
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke({"value": 0, "log": []})
        assert result["value"] == 1

    def test_interceptor_hooks_called(self):
        rec = RecordingInterceptor()
        g = InstrumentedGraph(SimpleState, interceptors=[rec])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke({"value": 0, "log": []})

        assert ("before", "inc") in rec.calls
        assert ("after", "inc") in rec.calls

    def test_interceptor_on_error(self):
        rec = RecordingInterceptor()
        g = InstrumentedGraph(SimpleState, interceptors=[rec])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError, match="node exploded"):
            compiled.invoke({"value": 0, "log": []})

        assert ("before", "fail") in rec.calls
        assert ("on_error", "fail") in rec.calls

    def test_broken_interceptor_does_not_crash(self):
        """Broken interceptor hooks are swallowed — node still executes."""
        g = InstrumentedGraph(SimpleState, interceptors=[BrokenInterceptor()])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke({"value": 5, "log": []})
        assert result["value"] == 6

    def test_middleware_transforms_result(self):
        g = InstrumentedGraph(SimpleState, middleware=[AddTagMiddleware()])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke({"value": 0, "log": []})
        assert result["value"] == 1
        assert "tagged:inc" in result["log"]

    def test_broken_middleware_propagates(self):
        """Middleware errors MUST propagate — silent corruption is worse."""
        g = InstrumentedGraph(SimpleState, middleware=[BrokenMiddleware()])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError, match="middleware exploded"):
            compiled.invoke({"value": 0, "log": []})

    def test_add_interceptor_after_construction(self):
        rec = RecordingInterceptor()
        g = InstrumentedGraph(SimpleState)
        g.add_interceptor(rec)
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke({"value": 0, "log": []})
        assert ("before", "inc") in rec.calls


class TestMetricsInterceptor:
    def test_collects_metrics(self):
        mi = MetricsInterceptor()
        g = InstrumentedGraph(SimpleState, interceptors=[mi])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke({"value": 0, "log": []})
        compiled.invoke({"value": 0, "log": []})

        assert "inc" in mi.metrics
        assert mi.metrics["inc"].call_count == 2
        assert mi.metrics["inc"].error_count == 0
        assert mi.metrics["inc"].total_duration > 0

    def test_snapshot_json_safe(self):
        mi = MetricsInterceptor()
        g = InstrumentedGraph(SimpleState, interceptors=[mi])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke({"value": 0, "log": []})

        snap = mi.snapshot()
        assert "inc" in snap
        assert snap["inc"]["call_count"] == 1
        assert isinstance(snap["inc"]["avg_duration"], float)

    def test_error_metrics(self):
        mi = MetricsInterceptor()
        g = InstrumentedGraph(SimpleState, interceptors=[mi])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError):
            compiled.invoke({"value": 0, "log": []})

        assert mi.metrics["fail"].call_count == 1
        assert mi.metrics["fail"].error_count == 1
