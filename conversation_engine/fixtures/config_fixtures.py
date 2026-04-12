"""
Shared configuration and state fixtures for tests and examples.

Provides reusable factories for:
- IntegrityRule instances (goal→req)
- DomainConfig instances (sample, partial, minimal)
- ConversationState dicts with sensible defaults
- ArchitecturalOntologyContext from various graph shapes
"""

from __future__ import annotations

from conversation_engine.graph.architectural_context import ArchitecturalOntologyContext
from conversation_engine.graph.state import ConversationState
from conversation_engine.infrastructure.llm.architectural_quiz import ARCHITECTURAL_QUIZ
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.project_spec import ProjectSpecification
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.snapshot_facade import graph_to_snapshot

# ── Rule factories ───────────────────────────────────────────────────


def goal_req_rule() -> IntegrityRule:
    """Every goal must have ≥1 SATISFIED_BY → requirement."""
    return IntegrityRule(
        id="rule-goal-req",
        name="Goal → Requirement",
        description="Every goal must have at least one requirement",
        applies_to_node_type="goal",
        rule_type="minimum_outgoing_edge_count",
        target_node_types=["requirement"],
        minimum_count=1,
        severity="high",
        failure_message_template="Goal '{subject_name}' has no requirements.",
    )


def req_step_rule() -> IntegrityRule:
    """Every requirement must have ≥1 REALIZED_BY → step."""
    return IntegrityRule(
        id="rule-req-step",
        name="Requirement → Step",
        description="Every requirement must have at least one step",
        applies_to_node_type="requirement",
        rule_type="minimum_outgoing_edge_count",
        target_node_types=["step"],
        minimum_count=1,
        severity="medium",
        failure_message_template="Requirement '{subject_name}' has no steps.",
    )


def standard_rules() -> list[IntegrityRule]:
    """The standard integrity rules used across tests and examples."""
    return [goal_req_rule(), req_step_rule()]


# ── DomainConfig factories ───────────────────────────────────────────


def sample_config(
    project_name: str = "test-project",
    graph: KnowledgeGraph | None = None,
    spec: ProjectSpecification | None = None,
    rules: list[IntegrityRule] | None = None,
    include_quiz: bool = True,
    system_prompt: str = "Test system prompt.",
) -> DomainConfig:
    """
    Full DomainConfig with spec, rules, quiz, and system prompt.

    If neither graph nor spec is provided, creates a minimal goal→requirement spec.
    If graph is provided, converts it to a spec via the facade.
    """
    if spec is None:
        if graph is None:
            from conversation_engine.models import Goal, Requirement
            from conversation_engine.models.base import BaseEdge

            g = KnowledgeGraph()
            g.add_node(Goal(id="g1", name="Goal 1", statement="A goal"))
            g.add_node(Requirement(id="r1", name="Req 1"))
            g.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="g1", target_id="r1"))
            graph = g
        spec = graph_to_snapshot(project_name, graph)

    return DomainConfig(
        project_name=project_name,
        project_spec=spec,
        quiz=list(ARCHITECTURAL_QUIZ) if include_quiz else None,
        rules=rules or [goal_req_rule()],
        system_prompt=system_prompt,
    )


def partial_config(
    project_name: str = "test-project",
    graph: KnowledgeGraph | None = None,
    spec: ProjectSpecification | None = None,
    rules: list[IntegrityRule] | None = None,
    system_prompt: str = "Test system prompt.",
) -> DomainConfig:
    """DomainConfig without quiz (triggers default ARCHITECTURAL_QUIZ fallback)."""
    if spec is None:
        if graph is None:
            from conversation_engine.models import Goal, Requirement
            from conversation_engine.models.base import BaseEdge

            g = KnowledgeGraph()
            g.add_node(Goal(id="g1", name="Goal 1", statement="A goal"))
            g.add_node(Requirement(id="r1", name="Req 1"))
            g.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="g1", target_id="r1"))
            graph = g
        spec = graph_to_snapshot(project_name, graph)

    return DomainConfig(
        project_name=project_name,
        project_spec=spec,
        rules=rules or [goal_req_rule()],
        system_prompt=system_prompt,
    )


# ── Context factories ────────────────────────────────────────────────


def make_context(
    graph: KnowledgeGraph,
    rules: list[IntegrityRule] | None = None,
    project_name: str = "test",
) -> ArchitecturalOntologyContext:
    """Build an ArchitecturalOntologyContext from a graph and optional rules."""
    spec = graph_to_snapshot(project_name, graph)
    config = DomainConfig(
        project_name=project_name,
        project_spec=spec,
        rules=rules or [goal_req_rule()],
    )
    return ArchitecturalOntologyContext(config)


# ── State factories ──────────────────────────────────────────────────


def minimal_state(**overrides) -> ConversationState:
    """Build a minimal ConversationState dict with sensible defaults."""
    state: ConversationState = {
        "context": None,
        "session_id": "test-session",
        "project_name": None,
        "project_store": None,
        "llm": None,
        "human": None,
        "tool_client": None,
        "findings": [],
        "messages": [],
        "current_turn": 0,
        "status": "running",
        "node_result": None,
        "preflight_passed": False,
    }
    state.update(overrides)
    return state


def make_state(
    graph: KnowledgeGraph,
    rules: list[IntegrityRule] | None = None,
) -> ConversationState:
    """Build a ConversationState with an ArchitecturalOntologyContext already set."""
    return minimal_state(
        context=make_context(graph, rules),
    )
