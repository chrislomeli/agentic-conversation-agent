"""
Integration tests: infrastructure wired into the conversation graph.

Covers:
- New NodeMiddleware system with build_conversation_graph
- Legacy interceptors still work (backwards compatibility)
- Middleware composition in real conversation runs
- ErrorHandlingMiddleware replaces handle_error node
- LLM injectable via state["llm"] (stub)
- Pre-flight LLM validation as a graph node
"""

import pytest

from conversation_engine.graph.context import (
    ConversationContext,
    Finding,
    ValidationResult,
)
from conversation_engine.graph.builder import (
    build_conversation_graph,
    MAX_TURNS,
)
from conversation_engine.infrastructure.interceptors import (
    MetricsInterceptor,
    LoggingInterceptor,
)
from conversation_engine.infrastructure.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    ErrorHandlingMiddleware,
)
from conversation_engine.infrastructure.llm import (
    CallLLM,
    LLMRequest,
    LLMResponse,
    call_llm_stub,
)
from conversation_engine.models.validation_quiz import FactualQuiz


# ── Fake contexts ───────────────────────────────────────────────────


class _CleanContext:
    """Context with no findings and no preflight quiz — graph exits in one pass."""

    def validate(self, prior_findings):
        return ValidationResult(findings=[])

    def format_finding_summary(self, findings):
        return "All clear."

    def get_domain_state(self):
        return {}

    @property
    def system_prompt(self) -> str:
        return "You are a test assistant."

    @property
    def preflight_quiz(self) -> list:
        return []


class _GappyContext:
    """Context that always returns one finding — graph loops to max turns."""

    def validate(self, prior_findings):
        resolved = [f for f in prior_findings if f.resolved]
        return ValidationResult(
            findings=resolved
            + [
                Finding(
                    id="gap-1",
                    finding_type="test_gap",
                    severity="high",
                    subject_ids=["x"],
                    message="Something is missing.",
                ),
            ]
        )

    def format_finding_summary(self, findings):
        return f"{len(findings)} issue(s) found."

    def get_domain_state(self):
        return {}

    @property
    def system_prompt(self) -> str:
        return "You are a test assistant."

    @property
    def preflight_quiz(self) -> list:
        return []


class _FailingContext:
    """Context whose validate() raises — tests error routing."""

    def validate(self, prior_findings):
        raise RuntimeError("context exploded")

    def format_finding_summary(self, findings):
        return ""

    def get_domain_state(self):
        return {}

    @property
    def system_prompt(self) -> str:
        return ""

    @property
    def preflight_quiz(self) -> list:
        return []


class _PreflightContext:
    """Context with a preflight quiz — for testing pre-flight node."""

    def __init__(self, system_prompt_text="The system has goal and requirement types."):
        self._system_prompt = system_prompt_text

    def validate(self, prior_findings):
        return ValidationResult(findings=[])

    def format_finding_summary(self, findings):
        return "All clear."

    def get_domain_state(self):
        return {}

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def preflight_quiz(self) -> list:
        return [
            FactualQuiz(
                question="What node types exist?",
                expected_answer="goal, requirement",
                weight=1.0,
            ),
        ]


def _make_state(ctx, llm=None):
    return {
        "context": ctx,
        "session_id": "infra-test",
        "llm": llm,
        "findings": [],
        "messages": [],
        "current_turn": 0,
        "status": "running",
        "node_result": None,
        "preflight_passed": False,
    }


# ── Tests ───────────────────────────────────────────────────────────


class TestInterceptorsInGraph:
    def test_metrics_collected_on_clean_run(self):
        """MetricsInterceptor collects data for all nodes during a clean run."""
        mi = MetricsInterceptor()
        graph = build_conversation_graph(interceptors=[mi])
        state = _make_state(_CleanContext())

        result = graph.invoke(state)

        snap = mi.snapshot()
        assert "preflight" in snap
        assert "validate" in snap
        assert "converse" in snap
        assert snap["validate"]["call_count"] >= 1
        assert snap["converse"]["call_count"] >= 1

    def test_metrics_on_looping_run(self):
        """MetricsInterceptor shows multiple calls when graph loops."""
        mi = MetricsInterceptor()
        graph = build_conversation_graph(interceptors=[mi])
        state = _make_state(_GappyContext())

        result = graph.invoke(state)

        snap = mi.snapshot()
        assert snap["validate"]["call_count"] == MAX_TURNS
        assert snap["converse"]["call_count"] == MAX_TURNS

    def test_logging_interceptor_no_crash(self):
        """LoggingInterceptor runs without crashing the graph."""
        graph = build_conversation_graph(interceptors=[LoggingInterceptor()])
        state = _make_state(_CleanContext())

        result = graph.invoke(state)
        assert result["current_turn"] == 1

    def test_multiple_interceptors(self):
        """Multiple interceptors compose correctly."""
        mi = MetricsInterceptor()
        graph = build_conversation_graph(
            interceptors=[LoggingInterceptor(), mi],
        )
        state = _make_state(_CleanContext())

        result = graph.invoke(state)

        assert "validate" in mi.snapshot()
        assert result["current_turn"] == 1


class TestErrorRoutingInGraph:
    def test_failing_context_propagates_without_error_mw(self):
        """Without ErrorHandlingMiddleware, exceptions propagate."""
        graph = build_conversation_graph()
        state = _make_state(_FailingContext())

        with pytest.raises(RuntimeError, match="context exploded"):
            graph.invoke(state)

    def test_failing_context_caught_by_error_middleware(self):
        """ErrorHandlingMiddleware catches the exception and sets status='error'."""
        mm = MetricsMiddleware()
        ehm = ErrorHandlingMiddleware(swallow_exceptions=True)
        graph = build_conversation_graph(node_middleware=[mm, ehm])
        state = _make_state(_FailingContext())

        result = graph.invoke(state)

        assert result["status"] == "error"
        assert result["node_result"].error.code == "NODE_EXCEPTION"
        assert "context exploded" in result["node_result"].error.message
        assert mm.metrics["validate"].call_count == 1


# ── New middleware integration tests ─────────────────────────────────


class TestMiddlewareInGraph:
    def test_metrics_collected_on_clean_run(self):
        """MetricsMiddleware collects data for all nodes during a clean run."""
        mm = MetricsMiddleware()
        graph = build_conversation_graph(node_middleware=[mm])
        state = _make_state(_CleanContext())

        result = graph.invoke(state)

        snap = mm.snapshot()
        assert "preflight" in snap
        assert "validate" in snap
        assert "converse" in snap
        assert snap["validate"]["call_count"] >= 1

    def test_metrics_on_looping_run(self):
        """MetricsMiddleware shows multiple calls when graph loops."""
        mm = MetricsMiddleware()
        graph = build_conversation_graph(node_middleware=[mm])
        state = _make_state(_GappyContext())

        result = graph.invoke(state)

        snap = mm.snapshot()
        assert snap["validate"]["call_count"] == MAX_TURNS
        assert snap["converse"]["call_count"] == MAX_TURNS

    def test_logging_middleware_no_crash(self):
        """LoggingMiddleware runs without crashing the graph."""
        graph = build_conversation_graph(node_middleware=[LoggingMiddleware()])
        state = _make_state(_CleanContext())

        result = graph.invoke(state)
        assert result["current_turn"] == 1

    def test_multiple_middleware_compose(self):
        """Multiple middleware compose correctly in the conversation graph."""
        mm = MetricsMiddleware()
        graph = build_conversation_graph(
            node_middleware=[LoggingMiddleware(), mm, ErrorHandlingMiddleware()],
        )
        state = _make_state(_CleanContext())

        result = graph.invoke(state)

        assert "validate" in mm.snapshot()
        assert result["current_turn"] == 1


class TestLLMInjection:
    def test_stub_llm_in_state(self):
        """State can carry an LLM callable used by the converse node."""
        graph = build_conversation_graph()
        state = _make_state(_CleanContext(), llm=call_llm_stub)

        result = graph.invoke(state)

        assert result["current_turn"] == 1
        assert result.get("llm") is call_llm_stub

    def test_none_llm_in_state(self):
        """State works fine without an LLM (backwards compat)."""
        graph = build_conversation_graph()
        state = _make_state(_CleanContext(), llm=None)

        result = graph.invoke(state)
        assert result["current_turn"] == 1


# ── Pre-flight LLM validation tests ─────────────────────────────────
# Preflight is now a graph node, not imperative build-time code.
# The quiz and system_prompt come from the ConversationContext.
# The LLM comes from state["llm"].


def _smart_llm(request: LLMRequest) -> LLMResponse:
    """Echoes system prompt — always contains the right concepts."""
    return LLMResponse(content=request.system_prompt, model="smart", success=True)


def _dumb_llm(request: LLMRequest) -> LLMResponse:
    """Always returns irrelevant text."""
    return LLMResponse(content="I like pizza.", model="dumb", success=True)


class TestPreflightNode:
    def test_smart_llm_passes_preflight(self):
        """Smart LLM passes preflight — graph continues to validate/reason/respond."""
        graph = build_conversation_graph()
        state = _make_state(_PreflightContext(), llm=_smart_llm)

        result = graph.invoke(state)

        assert result["preflight_passed"] is True
        assert result["current_turn"] == 1
        assert result["status"] != "error"

    def test_dumb_llm_fails_preflight(self):
        """Dumb LLM fails preflight — graph exits with status='error'."""
        graph = build_conversation_graph()
        state = _make_state(_PreflightContext(), llm=_dumb_llm)

        result = graph.invoke(state)

        assert result["preflight_passed"] is False
        assert result["status"] == "error"
        # Should not have reached validate/reason/respond
        assert result["current_turn"] == 0

    def test_no_quiz_skips_preflight(self):
        """Context with empty quiz — preflight passes through immediately."""
        graph = build_conversation_graph()
        state = _make_state(_CleanContext(), llm=_dumb_llm)

        result = graph.invoke(state)

        assert result["preflight_passed"] is True
        assert result["current_turn"] == 1

    def test_no_llm_skips_preflight(self):
        """No LLM in state — preflight passes through immediately."""
        graph = build_conversation_graph()
        state = _make_state(_PreflightContext(), llm=None)

        result = graph.invoke(state)

        assert result["preflight_passed"] is True
        assert result["current_turn"] == 1

    def test_preflight_runs_once(self):
        """Preflight only runs the LLM on the first turn, not on loop-back."""
        mm = MetricsMiddleware()
        graph = build_conversation_graph(node_middleware=[mm])

        # Use a gappy context WITH a quiz so the graph loops
        class _GappyWithQuiz(_GappyContext):
            @property
            def system_prompt(self) -> str:
                return "The system has goal and requirement types."

            @property
            def preflight_quiz(self) -> list:
                return [
                    FactualQuiz(
                        question="What types?",
                        expected_answer="goal, requirement",
                        weight=1.0,
                    ),
                ]

        state = _make_state(_GappyWithQuiz(), llm=_smart_llm)
        result = graph.invoke(state)

        # Preflight called once, but validate/converse loop MAX_TURNS
        snap = mm.snapshot()
        assert snap["preflight"]["call_count"] == 1
        assert snap["validate"]["call_count"] == MAX_TURNS

    def test_preflight_failure_message(self):
        """Failed preflight adds an informative message to state."""
        graph = build_conversation_graph()
        state = _make_state(_PreflightContext(), llm=_dumb_llm)

        result = graph.invoke(state)

        assert result["status"] == "error"
        assert len(result["messages"]) >= 1
        last_msg = result["messages"][-1]
        assert "pre-flight validation" in last_msg.content.lower()
