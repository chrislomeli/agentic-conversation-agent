"""
Tests for the composable NodeMiddleware system.

Covers all 7 middleware implementations:
  1. LoggingMiddleware — entry/exit/error logging
  2. MetricsMiddleware — call counts, durations, errors
  3. ValidationMiddleware — Pydantic schema validation
  4. ErrorHandlingMiddleware — exception catching + NodeResult failure detection
  5. RetryMiddleware — configurable retry with backoff
  6. CircuitBreakerMiddleware — CLOSED/OPEN/HALF_OPEN state machine
  7. ConfigMiddleware — per-node config injection

Also covers:
  - NodeMiddleware ABC: applies_to() per-node selectivity
  - InstrumentedGraph: chain building, ordering, composition
"""

from typing import TypedDict

import pytest
from langgraph.graph import END, START
from pydantic import BaseModel

from conversation_engine.infrastructure.instrumented_graph import InstrumentedGraph
from conversation_engine.infrastructure.middleware.base import NodeMiddleware
from conversation_engine.infrastructure.middleware.circuit_breaker_mw import (
    CircuitBreakerMiddleware,
    CircuitState,
)
from conversation_engine.infrastructure.middleware.config_mw import ConfigMiddleware
from conversation_engine.infrastructure.middleware.error_handling_mw import ErrorHandlingMiddleware
from conversation_engine.infrastructure.middleware.logging_mw import LoggingMiddleware
from conversation_engine.infrastructure.middleware.metrics_mw import MetricsMiddleware
from conversation_engine.infrastructure.middleware.retry_mw import RetryMiddleware
from conversation_engine.infrastructure.middleware.validation_mw import ValidationMiddleware
from conversation_engine.infrastructure.node_validation.result_schema import NodeResult

# ── Test fixtures ───────────────────────────────────────────────────


class SimpleState(TypedDict):
    value: int
    status: str
    node_result: NodeResult | None


def _state(value: int = 0, **extra) -> dict:
    """Build a consistent initial state dict."""
    return {"value": value, "status": "running", "node_result": None, **extra}


def increment(state: SimpleState) -> dict:
    return {"value": state["value"] + 1}


def failing_node(state: SimpleState) -> dict:
    raise RuntimeError("node exploded")


_call_count = 0


def flaky_node(state: SimpleState) -> dict:
    """Fails the first N times, then succeeds."""
    global _call_count
    _call_count += 1
    if _call_count <= state.get("_fail_count", 2):
        raise RuntimeError(f"flaky failure #{_call_count}")
    return {"value": state["value"] + 1}


class RecordingMiddleware(NodeMiddleware):
    """Records calls for assertion."""

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.calls: list[str] = []

    def __call__(self, node_name, state, next_fn):
        self.calls.append(f"before:{self.name}:{node_name}")
        result = next_fn(state)
        self.calls.append(f"after:{self.name}:{node_name}")
        return result


# ── NodeMiddleware ABC ──────────────────────────────────────────────


class TestNodeMiddlewareBase:
    def test_applies_to_all_by_default(self):
        mw = RecordingMiddleware("r")
        assert mw.applies_to("any_node")
        assert mw.applies_to("another_node")

    def test_applies_to_specific_nodes(self):
        mw = RecordingMiddleware("r", nodes={"reason", "validate"})
        assert mw.applies_to("reason")
        assert mw.applies_to("validate")
        assert not mw.applies_to("respond")

    def test_nodes_frozen_after_init(self):
        mw = RecordingMiddleware("r", nodes={"a"})
        # _nodes is a frozenset — immutable
        assert isinstance(mw._nodes, frozenset)


# ── Chain building in InstrumentedGraph ─────────────────────────────


class TestMiddlewareChain:
    def test_chain_executes_in_order(self):
        r1 = RecordingMiddleware("outer")
        r2 = RecordingMiddleware("inner")
        g = InstrumentedGraph(SimpleState, node_middleware=[r1, r2])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())

        # Outer wraps inner: outer.before → inner.before → [node] → inner.after → outer.after
        assert r1.calls == ["before:outer:inc", "after:outer:inc"]
        assert r2.calls == ["before:inner:inc", "after:inner:inc"]

    def test_per_node_selectivity_in_chain(self):
        all_nodes = RecordingMiddleware("all")
        inc_only = RecordingMiddleware("inc_only", nodes={"inc"})

        g = InstrumentedGraph(SimpleState, node_middleware=[all_nodes, inc_only])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())

        assert "before:all:inc" in all_nodes.calls
        assert "before:inc_only:inc" in inc_only.calls

    def test_graph_works_without_middleware(self):
        g = InstrumentedGraph(SimpleState)
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state(5))
        assert result["value"] == 6

    def test_add_node_middleware_after_construction(self):
        mm = MetricsMiddleware()
        g = InstrumentedGraph(SimpleState)
        g.add_node_middleware(mm)
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())
        assert "inc" in mm.metrics


# ── LoggingMiddleware ───────────────────────────────────────────────


class TestLoggingMiddleware:
    def test_no_crash_on_success(self):
        g = InstrumentedGraph(SimpleState, node_middleware=[LoggingMiddleware()])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state())
        assert result["value"] == 1

    def test_no_crash_on_error(self):
        g = InstrumentedGraph(SimpleState, node_middleware=[LoggingMiddleware()])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError, match="node exploded"):
            compiled.invoke(_state())

    def test_per_node_selectivity(self):
        lm = LoggingMiddleware(nodes={"inc"})
        assert lm.applies_to("inc")
        assert not lm.applies_to("other")


# ── MetricsMiddleware ───────────────────────────────────────────────


class TestMetricsMiddleware:
    def test_collects_metrics(self):
        mm = MetricsMiddleware()
        g = InstrumentedGraph(SimpleState, node_middleware=[mm])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())
        compiled.invoke(_state())

        assert mm.metrics["inc"].call_count == 2
        assert mm.metrics["inc"].error_count == 0
        assert mm.metrics["inc"].total_duration > 0

    def test_error_metrics(self):
        mm = MetricsMiddleware()
        g = InstrumentedGraph(SimpleState, node_middleware=[mm])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError):
            compiled.invoke(_state())

        assert mm.metrics["fail"].call_count == 1
        assert mm.metrics["fail"].error_count == 1

    def test_snapshot_json_safe(self):
        mm = MetricsMiddleware()
        g = InstrumentedGraph(SimpleState, node_middleware=[mm])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())

        snap = mm.snapshot()
        assert snap["inc"]["call_count"] == 1
        assert isinstance(snap["inc"]["avg_duration"], float)

    def test_per_node_selectivity(self):
        mm = MetricsMiddleware(nodes={"inc"})
        g = InstrumentedGraph(SimpleState, node_middleware=[mm])
        g.add_node("inc", increment)
        g.add_node("inc2", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", "inc2")
        g.add_edge("inc2", END)
        compiled = g.compile()

        compiled.invoke(_state())

        assert "inc" in mm.metrics
        assert "inc2" not in mm.metrics


# ── ValidationMiddleware ────────────────────────────────────────────


class ValidInput(BaseModel):
    model_config = {"extra": "ignore"}
    value: int
    status: str


class StrictInput(BaseModel):
    model_config = {"extra": "ignore"}
    value: int
    name: str  # required, not in SimpleState


class TestValidationMiddleware:
    def test_valid_input_passes(self):
        vm = ValidationMiddleware(schemas={"inc": ValidInput})
        g = InstrumentedGraph(SimpleState, node_middleware=[vm])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state(5))
        assert result["value"] == 6

    def test_invalid_input_returns_failure(self):
        vm = ValidationMiddleware(schemas={"inc": StrictInput})
        g = InstrumentedGraph(SimpleState, node_middleware=[vm])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state(5))
        # Should have a node_result failure (name is missing)
        assert result["node_result"] is not None
        assert not result["node_result"].ok
        assert result["node_result"].error.code == "INVALID_INPUT"

    def test_unregistered_node_passes_through(self):
        vm = ValidationMiddleware(schemas={"other_node": StrictInput})
        g = InstrumentedGraph(SimpleState, node_middleware=[vm])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state(5))
        assert result["value"] == 6


# ── ErrorHandlingMiddleware ─────────────────────────────────────────


class TestErrorHandlingMiddleware:
    def test_swallows_exception(self):
        ehm = ErrorHandlingMiddleware(swallow_exceptions=True)
        g = InstrumentedGraph(SimpleState, node_middleware=[ehm])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        result = compiled.invoke(_state())
        assert result["status"] == "error"
        assert result["node_result"].error.code == "NODE_EXCEPTION"
        assert "node exploded" in result["node_result"].error.message

    def test_propagates_exception_when_configured(self):
        ehm = ErrorHandlingMiddleware(swallow_exceptions=False)
        g = InstrumentedGraph(SimpleState, node_middleware=[ehm])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError, match="node exploded"):
            compiled.invoke(_state())

    def test_detects_node_result_failure(self):
        def returns_failure(state):
            return {"node_result": NodeResult.failure("BAD", "something bad")}

        ehm = ErrorHandlingMiddleware()
        g = InstrumentedGraph(SimpleState, node_middleware=[ehm])
        g.add_node("bad", returns_failure)
        g.add_edge(START, "bad")
        g.add_edge("bad", END)
        compiled = g.compile()

        result = compiled.invoke(_state())
        assert result["status"] == "error"

    def test_success_passes_through(self):
        ehm = ErrorHandlingMiddleware()
        g = InstrumentedGraph(SimpleState, node_middleware=[ehm])
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state())
        assert result["value"] == 1

    def test_per_node_selectivity(self):
        ehm = ErrorHandlingMiddleware(nodes={"fail"}, swallow_exceptions=True)
        g = InstrumentedGraph(SimpleState, node_middleware=[ehm])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        # "fail" is covered — exception swallowed
        result = compiled.invoke(_state())
        assert result["status"] == "error"


# ── RetryMiddleware ─────────────────────────────────────────────────


class TestRetryMiddleware:
    def test_retries_on_failure(self):
        global _call_count
        _call_count = 0

        rm = RetryMiddleware(max_retries=3, backoff_base=0.0, nodes={"flaky"})
        g = InstrumentedGraph(SimpleState, node_middleware=[rm])
        g.add_node("flaky", flaky_node)
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        compiled = g.compile()

        result = compiled.invoke(_state(10, _fail_count=2))
        assert result["value"] == 11
        assert _call_count == 3  # 2 failures + 1 success

    def test_exhausts_retries(self):
        global _call_count
        _call_count = 0

        rm = RetryMiddleware(max_retries=1, backoff_base=0.0, nodes={"flaky"})
        g = InstrumentedGraph(SimpleState, node_middleware=[rm])
        g.add_node("flaky", flaky_node)
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        compiled = g.compile()

        with pytest.raises(RuntimeError, match="flaky failure"):
            compiled.invoke(_state(_fail_count=5))

    def test_no_retry_on_success(self):
        global _call_count
        _call_count = 100  # Already past the fail count

        rm = RetryMiddleware(max_retries=3, backoff_base=0.0)
        g = InstrumentedGraph(SimpleState, node_middleware=[rm])
        g.add_node("flaky", flaky_node)
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        compiled = g.compile()

        result = compiled.invoke(_state(_fail_count=0))
        assert result["value"] == 1

    def test_selective_error_types(self):
        rm = RetryMiddleware(
            max_retries=2,
            retryable_errors=(ValueError,),
            backoff_base=0.0,
        )

        def raises_type_error(state):
            raise TypeError("not retryable")

        g = InstrumentedGraph(SimpleState, node_middleware=[rm])
        g.add_node("te", raises_type_error)
        g.add_edge(START, "te")
        g.add_edge("te", END)
        compiled = g.compile()

        # TypeError is NOT in retryable_errors → should raise immediately
        with pytest.raises(TypeError, match="not retryable"):
            compiled.invoke(_state())

    def test_skips_non_target_nodes(self):
        global _call_count
        _call_count = 0

        rm = RetryMiddleware(max_retries=3, backoff_base=0.0, nodes={"other"})
        g = InstrumentedGraph(SimpleState, node_middleware=[rm])
        g.add_node("fail", failing_node)
        g.add_edge(START, "fail")
        g.add_edge("fail", END)
        compiled = g.compile()

        # "fail" is NOT in nodes — no retry, exception propagates immediately
        with pytest.raises(RuntimeError, match="node exploded"):
            compiled.invoke(_state())


# ── CircuitBreakerMiddleware ────────────────────────────────────────


class TestCircuitBreakerMiddleware:
    def test_starts_closed(self):
        cb = CircuitBreakerMiddleware(failure_threshold=3)
        assert cb.get_state("any") == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=2,
            cooldown_seconds=60.0,
            nodes={"fail"},
        )

        for _ in range(2):
            try:
                cb("fail", {}, failing_node)
            except RuntimeError:
                pass

        assert cb.get_state("fail") == CircuitState.OPEN

    def test_short_circuits_when_open(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=1,
            cooldown_seconds=60.0,
            nodes={"fail"},
        )

        # Trip the circuit
        try:
            cb("fail", {}, failing_node)
        except RuntimeError:
            pass

        assert cb.get_state("fail") == CircuitState.OPEN

        # Next call should be short-circuited, not raise
        result = cb("fail", {"value": 0, "status": "running"}, failing_node)
        assert result["status"] == "error"
        assert result["node_result"].error.code == "CIRCUIT_OPEN"

    def test_half_open_after_cooldown(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=1,
            cooldown_seconds=0.0,  # Instant cooldown for testing
            success_threshold=1,
            nodes={"fail"},
        )

        # Trip the circuit
        try:
            cb("fail", {}, failing_node)
        except RuntimeError:
            pass

        assert cb.get_state("fail") == CircuitState.OPEN

        # With 0s cooldown, next call transitions to HALF_OPEN and executes
        # Since the node still fails, it stays open
        try:
            cb("fail", {}, failing_node)
        except RuntimeError:
            pass

        assert cb.get_state("fail") == CircuitState.OPEN

    def test_closes_on_success_in_half_open(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=1,
            cooldown_seconds=0.0,
            success_threshold=1,
            nodes={"n"},
        )

        # Trip the circuit
        try:
            cb("n", {}, failing_node)
        except RuntimeError:
            pass

        assert cb.get_state("n") == CircuitState.OPEN

        # With 0s cooldown, allow a success through
        result = cb("n", {"value": 0}, increment)
        assert result["value"] == 1
        assert cb.get_state("n") == CircuitState.CLOSED

    def test_resets_on_success_when_closed(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=3,
            nodes={"n"},
        )

        # One failure
        try:
            cb("n", {}, failing_node)
        except RuntimeError:
            pass

        assert cb._circuits["n"].failure_count == 1

        # One success resets
        cb("n", {"value": 0}, increment)
        assert cb._circuits["n"].failure_count == 0

    def test_skips_non_target_nodes(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=1,
            nodes={"other"},
        )

        # "fail" is not in nodes — passes through directly
        with pytest.raises(RuntimeError, match="node exploded"):
            cb("fail", {}, failing_node)

        # No circuit created for "fail"
        assert "fail" not in cb._circuits


# ── ConfigMiddleware ────────────────────────────────────────────────


class TestConfigMiddleware:
    def test_injects_config(self):
        received_config = {}

        def capture_config(state):
            received_config.update(state.get("_node_config", {}))
            return {"value": state["value"] + 1}

        cm = ConfigMiddleware(
            config={
                "inc": {"temperature": 0.5, "model": "gpt-4"},
            }
        )
        g = InstrumentedGraph(SimpleState, node_middleware=[cm])
        g.add_node("inc", capture_config)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())
        assert received_config == {"temperature": 0.5, "model": "gpt-4"}

    def test_empty_config_for_unconfigured_node(self):
        received_config = {"sentinel": True}

        def capture_config(state):
            received_config.clear()
            received_config.update(state.get("_node_config", {}))
            return {"value": state["value"] + 1}

        cm = ConfigMiddleware(config={"other": {"x": 1}})
        g = InstrumentedGraph(SimpleState, node_middleware=[cm])
        g.add_node("inc", capture_config)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())
        assert received_config == {}

    def test_custom_config_key(self):
        received = {}

        def capture(state):
            received.update(state.get("my_cfg", {}))
            return {"value": state["value"] + 1}

        cm = ConfigMiddleware(
            config={"inc": {"key": "value"}},
            config_key="my_cfg",
        )
        g = InstrumentedGraph(SimpleState, node_middleware=[cm])
        g.add_node("inc", capture)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())
        assert received == {"key": "value"}


# ── Composition tests ───────────────────────────────────────────────


class TestMiddlewareComposition:
    def test_logging_plus_metrics(self):
        """Logging + Metrics compose without interference."""
        mm = MetricsMiddleware()
        g = InstrumentedGraph(
            SimpleState,
            node_middleware=[LoggingMiddleware(), mm],
        )
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        compiled.invoke(_state())

        assert mm.metrics["inc"].call_count == 1

    def test_error_handling_plus_retry(self):
        """ErrorHandling outside Retry: retry happens, then error handling catches."""
        global _call_count
        _call_count = 0

        ehm = ErrorHandlingMiddleware(swallow_exceptions=True)
        rm = RetryMiddleware(max_retries=1, backoff_base=0.0)

        g = InstrumentedGraph(
            SimpleState,
            node_middleware=[ehm, rm],  # Error wraps Retry wraps node
        )
        g.add_node("flaky", flaky_node)
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        compiled = g.compile()

        # _fail_count=5 → both retries fail → error handling catches
        result = compiled.invoke(_state(_fail_count=5))
        assert result["status"] == "error"
        assert result["node_result"].error.code == "NODE_EXCEPTION"

    def test_validation_plus_error_handling(self):
        """Validation fails → ErrorHandling detects NodeResult failure."""
        vm = ValidationMiddleware(schemas={"inc": StrictInput})
        ehm = ErrorHandlingMiddleware()

        g = InstrumentedGraph(
            SimpleState,
            node_middleware=[ehm, vm],  # Error wraps Validation wraps node
        )
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state())
        assert result["status"] == "error"
        assert result["node_result"].error.code == "INVALID_INPUT"

    def test_full_stack(self):
        """All middleware compose in a realistic order."""
        mm = MetricsMiddleware()

        g = InstrumentedGraph(
            SimpleState,
            node_middleware=[
                LoggingMiddleware(),
                mm,
                ErrorHandlingMiddleware(swallow_exceptions=True),
                ConfigMiddleware(config={"inc": {"debug": True}}),
            ],
        )
        g.add_node("inc", increment)
        g.add_edge(START, "inc")
        g.add_edge("inc", END)
        compiled = g.compile()

        result = compiled.invoke(_state())
        assert result["value"] == 1
        assert mm.metrics["inc"].call_count == 1
