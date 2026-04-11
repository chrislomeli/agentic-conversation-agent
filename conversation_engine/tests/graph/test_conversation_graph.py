"""
Tests for the conversation graph.

Covers:
- State schema basics
- validate node (produces findings from violations via context)
- converse node (LLM + human exchange, bumps turn counter)
- Full graph: complete graph → single pass, exits clean
- Full graph: graph with gaps → loops, finds violations
- Full graph: max turns guard
- Router logic
- Domain-agnosticism: fake context proves the loop has no domain imports
"""

from conversation_engine.graph.context import (
    ConversationContext,
    Finding,
    ValidationResult,
)
from conversation_engine.graph.nodes import validate, converse
from conversation_engine.graph.builder import (
    build_conversation_graph,
    route_after_converse,
    MAX_TURNS,
)
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.fixtures import (
    goal_req_rule,
    req_step_rule,
    make_context,
    make_state,
    create_graph_with_gaps,
    create_graph_complete,
    create_minimal_graph,
)


# ── Node unit tests ─────────────────────────────────────────────────

class TestValidateNode:
    """Test the validate node in isolation."""

    def test_complete_graph_no_findings(self):
        """Complete graph produces no open findings."""
        state = make_state(create_graph_complete())
        result = validate(state)

        assert result["status"] == "running"
        assert len(result["findings"]) == 0

    def test_graph_with_gaps_produces_findings(self):
        """Graph with gaps produces findings for each violation."""
        state = make_state(create_graph_with_gaps())
        result = validate(state)

        assert len(result["findings"]) > 0
        # All new findings should be unresolved
        for f in result["findings"]:
            assert not f.resolved

    def test_resolved_findings_preserved(self):
        """Previously resolved findings are kept across re-validation."""
        resolved = Finding(
            id="old-resolved",
            finding_type="missing_goal_coverage",
            severity="high",
            subject_ids=["goal-99"],
            message="Already fixed",
            resolved=True,
            conversation_turn_id="turn-1",
        )
        state = make_state(create_graph_with_gaps())
        state["findings"] = [resolved]

        result = validate(state)

        ids = [f.session_id for f in result["findings"]]
        assert "old-resolved" in ids


class TestConverseNode:
    """Test the converse node in isolation (no LLM, no human)."""

    def test_no_findings_positive_message(self):
        """When no findings, converse gives a positive summary."""
        state = make_state(create_graph_complete())
        state["findings"] = []

        result = converse(state)

        assert len(result["messages"]) == 1
        assert "complete" in result["messages"][0].content.lower() or \
               "pass" in result["messages"][0].content.lower()

    def test_with_findings_lists_issues(self):
        """When findings exist, converse summarises them."""
        state = make_state(create_graph_with_gaps())
        state["findings"] = [
            Finding(
                id="f1",
                finding_type="missing_goal_coverage",
                severity="high",
                subject_ids=["goal-2"],
                message="Goal 'Orphan Goal' has no requirements.",
            ),
        ]

        result = converse(state)

        content = result["messages"][0].content
        assert "1 issue" in content
        assert "high" in content.lower()

    def test_increments_turn(self):
        state = make_state(create_minimal_graph())
        state["current_turn"] = 2

        result = converse(state)
        assert result["current_turn"] == 3

    def test_starts_at_zero(self):
        state = make_state(create_minimal_graph())

        result = converse(state)
        assert result["current_turn"] == 1


# ── Router tests ─────────────────────────────────────────────────────

class TestRouter:
    """Test route_after_converse logic."""

    def test_exits_when_no_open_findings(self):
        state = make_state(create_graph_complete())
        state["findings"] = []
        state["current_turn"] = 1

        assert route_after_converse(state) == "__end__"

    def test_loops_when_open_findings_remain(self):
        state = make_state(create_graph_with_gaps())
        state["findings"] = [
            Finding(
                id="f1",
                finding_type="missing_goal_coverage",
                severity="high",
                subject_ids=["goal-2"],
                message="Gap found",
            ),
        ]
        state["current_turn"] = 1

        assert route_after_converse(state) == "validate"

    def test_exits_on_max_turns(self):
        state = make_state(create_graph_with_gaps())
        state["findings"] = [
            Finding(
                id="f1",
                finding_type="missing_goal_coverage",
                severity="high",
                subject_ids=["goal-2"],
                message="Gap found",
            ),
        ]
        state["current_turn"] = MAX_TURNS

        assert route_after_converse(state) == "__end__"

    def test_exits_on_complete_status(self):
        state = make_state(create_graph_complete())
        state["status"] = "complete"

        assert route_after_converse(state) == "__end__"

    def test_exits_on_error_status(self):
        state = make_state(create_graph_complete())
        state["status"] = "error"

        assert route_after_converse(state) == "__end__"


# ── Full graph integration tests ─────────────────────────────────────

class TestConversationGraphIntegration:
    """Test the compiled conversation graph end-to-end."""

    def test_complete_graph_single_pass(self):
        """A complete graph should exit after one pass with no issues."""
        graph = build_conversation_graph()
        state = make_state(create_graph_complete())

        result = graph.invoke(state)

        # Should have no open findings
        open_findings = [f for f in result["findings"] if not f.resolved]
        assert len(open_findings) == 0
        # Should have produced at least one AI message
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        assert len(ai_messages) >= 1

    def test_graph_with_gaps_finds_violations(self):
        """A graph with gaps should produce findings."""
        graph = build_conversation_graph()
        state = make_state(create_graph_with_gaps())

        result = graph.invoke(state)

        # Should exit (max turns) with open findings still present
        # because nobody resolves them in this stub
        assert result["current_turn"] == MAX_TURNS
        open_findings = [f for f in result["findings"] if not f.resolved]
        assert len(open_findings) > 0

    def test_multiple_rules(self):
        """Test graph with multiple rules finds violations from each."""
        graph = build_conversation_graph()
        state = make_state(
            create_graph_with_gaps(),
            rules=[goal_req_rule(), req_step_rule()],
        )

        result = graph.invoke(state)

        # Should find violations from both rules
        open_findings = [f for f in result["findings"] if not f.resolved]
        rule_ids = set()
        for f in open_findings:
            rule_ids.update(f.related_rule_ids)
        assert "rule-goal-req" in rule_ids
        assert "rule-req-step" in rule_ids

    def test_minimal_graph_passes(self):
        """Minimal graph with matching rule should pass."""
        graph = build_conversation_graph()
        state = make_state(create_minimal_graph())

        result = graph.invoke(state)

        open_findings = [f for f in result["findings"] if not f.resolved]
        assert len(open_findings) == 0

    def test_turn_counter_advances(self):
        """Turn counter should advance each loop iteration."""
        graph = build_conversation_graph()
        state = make_state(create_graph_with_gaps())

        result = graph.invoke(state)

        assert result["current_turn"] == MAX_TURNS

    def test_session_id_preserved(self):
        """Session ID should pass through unchanged."""
        graph = build_conversation_graph()
        state = make_state(create_graph_complete())
        state["session_id"] = "my-unique-session"

        result = graph.invoke(state)

        assert result["session_id"] == "my-unique-session"


# ── Domain-agnostic tests ────────────────────────────────────────────

class _FakeContext:
    """
    A completely fake ConversationContext that has nothing to do with
    the architectural ontology.  Proves the loop is domain-agnostic.
    """

    def __init__(self, findings: list[Finding] | None = None):
        self._findings = findings or []

    def validate(self, prior_findings: list[Finding]) -> ValidationResult:
        resolved = [f for f in prior_findings if f.resolved]
        return ValidationResult(findings=resolved + self._findings)

    def format_finding_summary(self, findings: list[Finding]) -> str:
        if not findings:
            return "Everything looks good."
        return f"{len(findings)} problem(s) detected."

    def get_domain_state(self) -> dict:
        return {"domain": "fake"}

    @property
    def system_prompt(self) -> str:
        return "You are a fake test assistant."

    @property
    def preflight_quiz(self) -> list:
        return []


class TestDomainAgnosticism:
    """Prove the conversation loop works with a non-architectural context."""

    def test_fake_context_no_findings(self):
        """Loop exits cleanly with a fake context that finds nothing."""
        graph = build_conversation_graph()
        state = {
            "context": _FakeContext(),
            "session_id": "fake-session",
            "findings": [],
            "messages": [],
            "current_turn": 0,
            "status": "running",
            "preflight_passed": False,
        }

        result = graph.invoke(state)

        open_findings = [f for f in result["findings"] if not f.resolved]
        assert len(open_findings) == 0
        assert result["current_turn"] == 1  # single pass

    def test_fake_context_with_findings_loops(self):
        """Loop iterates when fake context produces unresolved findings."""
        findings = [
            Finding(
                id="fake-1",
                finding_type="lint_error",
                severity="medium",
                subject_ids=["file.py"],
                message="Unused import on line 3.",
            ),
        ]
        graph = build_conversation_graph()
        state = {
            "context": _FakeContext(findings=findings),
            "session_id": "fake-session",
            "findings": [],
            "messages": [],
            "current_turn": 0,
            "status": "running",
            "preflight_passed": False,
        }

        result = graph.invoke(state)

        assert result["current_turn"] == MAX_TURNS
        open_findings = [f for f in result["findings"] if not f.resolved]
        assert len(open_findings) > 0

    def test_fake_context_satisfies_protocol(self):
        """_FakeContext is recognized as a ConversationContext."""
        assert isinstance(_FakeContext(), ConversationContext)
