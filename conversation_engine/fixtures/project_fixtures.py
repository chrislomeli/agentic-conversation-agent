"""
ProjectSpecification fixtures for using this repo as the subject system.2OThis is the canonical dogfood spec: we use the project itself as the first
real-world example of what the system is meant to manage.

Design notes:
  - Goals reflect what we want the system to *do* for its user, not how it works.
  - Requirements are system needs, not implementation choices.
  - Steps are work items (coarse-grained, status-tracked), not file paths.
  - include_steps=True adds the current state of known work items.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.project_spec import (
    ConstraintSpec,
    DependencySpec,
    GoalSpec,
    ProjectSpecification,
    RequirementSpec,
    StepSpec,
)
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.models.validation_quiz import FactualQuiz


def conversation_engine_meta_spec(
    *,
    include_steps: bool = False,
) -> ProjectSpecification:
    """
    Canonical fixture: this repository as a project under management.

    Three goals:
      1. Persistent Project Context  — LLM picks up where it left off
      2. Constrained State Format    — structured, validatable, not free prose
      3. Progressive Demo Platform   — composable, teachable end-to-end
    """

    spec = ProjectSpecification(
        project_name="langgraph-ogar",
        description=(
            "A LangGraph-based conversation engine that maintains structured "
            "project context (goals, requirements, steps, constraints) so an LLM "
            "can pick up any project where it left off.  Internally uses a typed "
            "KnowledgeGraph with integrity rules; externally exposes only a flat, "
            "name-based ProjectSpecification.  Operates in two modes: as a "
            "standalone planning partner (the spec is the deliverable) or as a "
            "context provider for external tools like Claude Code via MCP."
        ),
        goals=[
            GoalSpec(
                name="Persistent Project Context",
                statement=(
                    "An LLM should be able to pick up a project conversation "
                    "where it left off — knowing what has been decided, what is "
                    "in progress, and what comes next — without relying on raw "
                    "conversation history."
                ),
            ),
            GoalSpec(
                name="Constrained State Format",
                statement=(
                    "Project status must be represented in a machine-readable, "
                    "schema-enforced format so the LLM cannot misinterpret or "
                    "hallucinate project state."
                ),
            ),
            GoalSpec(
                name="Progressive Demo Platform",
                statement=(
                    "The system should compose into progressively richer demos "
                    "that a professional would respect and a tutor could use to "
                    "teach incrementally — starting from minimal and building up."
                ),
            ),
        ],
        requirements=[
            # ── Goal 1: Persistent Project Context ──────────────────────────
            RequirementSpec(
                name="Session Continuity",
                goal_ref="Persistent Project Context",
                requirement_type="functional",
                description=(
                    "At the start of each conversation the agent loads the current "
                    "ProjectSpecification so it has full project context without "
                    "re-establishing it from scratch."
                ),
            ),
            RequirementSpec(
                name="Explicit Progress Tracking",
                goal_ref="Persistent Project Context",
                requirement_type="functional",
                description=(
                    "Steps carry explicit status (pending, in_progress, done, blocked) "
                    "and completion percentage.  Requirement completion is derived: "
                    "a requirement is satisfied when all its steps are done."
                ),
            ),
            RequirementSpec(
                name="Decision Memory",
                goal_ref="Persistent Project Context",
                requirement_type="functional",
                description=(
                    "Key decisions and their rationale are persisted as first-class "
                    "project artifacts so future sessions can see why things were "
                    "built the way they were."
                ),
            ),
            # ── Goal 2: Constrained State Format ────────────────────────────
            RequirementSpec(
                name="Schema-Enforced Project Spec",
                goal_ref="Constrained State Format",
                requirement_type="functional",
                description=(
                    "ProjectSpecification is the only structure the LLM interacts "
                    "with.  It uses human-readable names with no internal graph IDs "
                    "or edge types."
                ),
            ),
            RequirementSpec(
                name="Integrity Rule Validation",
                goal_ref="Constrained State Format",
                requirement_type="functional",
                description=(
                    "Declarative integrity rules define what structural completeness "
                    "means (e.g. every goal has requirements).  The system validates "
                    "the spec and reports violations as Findings."
                ),
            ),
            RequirementSpec(
                name="Tool-Governed Mutations",
                goal_ref="Constrained State Format",
                requirement_type="functional",
                description=(
                    "All changes to project state go through typed tool calls with "
                    "schema validation.  The LLM cannot modify state by producing "
                    "free-text prose."
                ),
            ),
            RequirementSpec(
                name="Durable Persistence",
                goal_ref="Constrained State Format",
                requirement_type="functional",
                description=(
                    "The KnowledgeGraph can be persisted to and reloaded from a "
                    "graph database (Memgraph or Neo4j) so project state survives "
                    "process restarts."
                ),
            ),
            # ── Goal 3: Progressive Demo Platform ───────────────────────────
            RequirementSpec(
                name="Minimal Viable Composition",
                goal_ref="Progressive Demo Platform",
                requirement_type="functional",
                description=(
                    "A new user can run a useful conversation with only goals and "
                    "requirements — steps, rules, and quiz are all optional add-ons."
                ),
            ),
            RequirementSpec(
                name="Golden Demo Path",
                goal_ref="Progressive Demo Platform",
                requirement_type="functional",
                description=(
                    "One polished end-to-end example exists: load context, identify "
                    "gaps, propose next actions, update progress, persist state."
                ),
            ),
            RequirementSpec(
                name="Teachable Increments",
                goal_ref="Progressive Demo Platform",
                requirement_type="functional",
                description=(
                    "Each capability (graph, validation, LLM loop, tools, persistence) "
                    "can be introduced independently so a tutor can build up the "
                    "system feature by feature."
                ),
            ),
        ],
        constraints=[
            ConstraintSpec(
                name="No Raw Graph Exposure to LLM",
                statement=(
                    "The LLM interacts only with ProjectSpecification-level structures. "
                    "Internal node IDs, edge types, and graph topology are never "
                    "surfaced in prompts or tool responses."
                ),
            ),
            ConstraintSpec(
                name="Steps Are Optional",
                statement=(
                    "A spec with only goals and requirements is valid and useful. "
                    "Steps add resolution but must not be required for the system "
                    "to function."
                ),
            ),
            ConstraintSpec(
                name="Dual-Mode Operation",
                statement=(
                    "The system must work as both a standalone planning partner "
                    "(the validated spec is the deliverable) and a context provider "
                    "for external tools (e.g. Claude Code via MCP) without requiring "
                    "different configurations or data models."
                ),
            ),
        ],
        dependencies=[
            DependencySpec(
                name="LangGraph",
                description="Conversation loop orchestration and state management.",
            ),
            DependencySpec(
                name="OpenAI / Anthropic API",
                description="LLM calls for reasoning, validation, and proposal generation.",
            ),
            DependencySpec(
                name="Pydantic",
                description="Schema enforcement for all spec and model types.",
            ),
            DependencySpec(
                name="Memgraph / Neo4j",
                description=(
                    "Graph database backend for durable KnowledgeGraph persistence. "
                    "Not yet integrated — in-memory store is the current fallback."
                ),
            ),
        ],
    )

    if include_steps:
        spec.steps = [
            # ── Done ────────────────────────────────────────────────────────
            StepSpec(
                name="KnowledgeGraph Core",
                requirement_refs=["Schema-Enforced Project Spec", "Integrity Rule Validation"],
                has_no_dependencies=False,
                dependency_refs=["Pydantic"],
                status="done",
                percentage=100,
                description=(
                    "In-memory directed typed graph with O(1) node/edge lookups "
                    "and indexed queries (neighbors, orphans, paths)."
                ),
            ),
            StepSpec(
                name="Snapshot Facade",
                requirement_refs=["Schema-Enforced Project Spec"],
                has_no_dependencies=False,
                dependency_refs=["KnowledgeGraph Core"],
                status="done",
                percentage=100,
                description=(
                    "Bidirectional conversion between flat ProjectSpecification "
                    "and KnowledgeGraph.  LLM sees names; graph sees IDs."
                ),
            ),
            StepSpec(
                name="LangGraph Conversation Loop",
                requirement_refs=["Session Continuity", "Tool-Governed Mutations"],
                has_no_dependencies=False,
                dependency_refs=["LangGraph", "KnowledgeGraph Core"],
                status="done",
                percentage=100,
                description=(
                    "Domain-agnostic preflight → validate → converse loop with "
                    "MAX_TURNS guard.  All domain logic injected via ConversationContext protocol."
                ),
            ),
            StepSpec(
                name="LLM Protocol Layer",
                requirement_refs=["Session Continuity"],
                has_no_dependencies=False,
                dependency_refs=["OpenAI / Anthropic API"],
                status="done",
                percentage=100,
                description=(
                    "CallLLM protocol + OpenAI adapter + deterministic stub.  "
                    "Swappable without graph rewrites."
                ),
            ),
            StepSpec(
                name="Integrity Rule Evaluator",
                requirement_refs=["Integrity Rule Validation"],
                has_no_dependencies=False,
                dependency_refs=["KnowledgeGraph Core"],
                status="done",
                percentage=100,
                description=(
                    "Deterministic (no LLM) evaluation of declarative IntegrityRules "
                    "against the graph.  Returns typed RuleViolation list."
                ),
            ),
            StepSpec(
                name="Tool Client Infrastructure",
                requirement_refs=["Tool-Governed Mutations"],
                has_no_dependencies=False,
                dependency_refs=["LangGraph"],
                status="done",
                percentage=100,
                description=(
                    "ReAct agent loop, typed tool envelope, transport-agnostic ToolClient contract."
                ),
            ),
            # ── In Progress ─────────────────────────────────────────────────
            StepSpec(
                name="Project Fixture (Dogfood Spec)",
                requirement_refs=["Minimal Viable Composition", "Golden Demo Path"],
                has_no_dependencies=True,
                status="in_progress",
                percentage=80,
                description=(
                    "Write the canonical ProjectSpecification for this repo itself "
                    "as the first real-world dogfood example."
                ),
            ),
            # ── Pending ─────────────────────────────────────────────────────
            StepSpec(
                name="Graph Database Persistence",
                requirement_refs=["Durable Persistence"],
                has_no_dependencies=False,
                dependency_refs=["Memgraph / Neo4j", "KnowledgeGraph Core"],
                status="pending",
                percentage=0,
                description=(
                    "Implement Memgraph/Neo4j backend that loads/saves the "
                    "KnowledgeGraph.  Replace InMemoryProjectStore."
                ),
            ),
            StepSpec(
                name="Session Reload on Conversation Start",
                requirement_refs=["Session Continuity"],
                has_no_dependencies=False,
                dependency_refs=["Graph Database Persistence", "Snapshot Facade"],
                status="pending",
                percentage=0,
                description=(
                    "At conversation start, load the saved ProjectSpecification and "
                    "inject it into ConversationContext so the LLM has full project "
                    "context immediately."
                ),
            ),
            StepSpec(
                name="Project Mutation Tools",
                requirement_refs=["Tool-Governed Mutations"],
                has_no_dependencies=False,
                dependency_refs=["Tool Client Infrastructure", "Snapshot Facade"],
                status="pending",
                percentage=0,
                description=(
                    "Typed tool implementations (add_goal, add_requirement, add_step, "
                    "update_step_status, etc.) that the LLM calls to mutate the "
                    "ProjectSpecification through the existing ToolClient contract."
                ),
            ),
            StepSpec(
                name="Non-Meta Demo Spec",
                requirement_refs=["Golden Demo Path", "Teachable Increments"],
                has_no_dependencies=False,
                dependency_refs=["Project Fixture (Dogfood Spec)"],
                status="pending",
                percentage=0,
                description=(
                    "A real-world example ProjectSpecification (e.g. mobile app "
                    "launch, microservices migration, course design) that tells a "
                    "story a stranger would care about.  The dogfood spec stays "
                    "as the internal development fixture."
                ),
            ),
            StepSpec(
                name="Golden Demo Example",
                requirement_refs=["Golden Demo Path", "Teachable Increments"],
                has_no_dependencies=False,
                dependency_refs=[
                    "Non-Meta Demo Spec",
                    "Project Mutation Tools",
                    "Session Reload on Conversation Start",
                    "LangGraph Conversation Loop",
                ],
                status="pending",
                percentage=0,
                description=(
                    "One polished runnable script using the non-meta demo spec: "
                    "load context, identify gaps, propose next actions via tool "
                    "calls, update progress, persist state."
                ),
            ),
        ]

    return spec


# ── Project-specific system prompt ─────────────────────────────────────

META_SYSTEM_PROMPT = """\
You are a project planning assistant for "langgraph-ogar", a conversation \
engine that maintains structured project context for LLMs.

## What this project is

A LangGraph-based system that lets an LLM pick up any project where it \
left off.  Internally it uses a typed KnowledgeGraph with integrity rules; \
externally it exposes only a flat, name-based ProjectSpecification.

## Data model

The project is described by three layers:
- **Goals**: desired outcomes.  Stable, rarely change.
- **Requirements**: system needs linked to goals.  No status field — \
  requirement satisfaction is *derived* from its steps.
- **Steps**: buildable work items with explicit status \
  (pending, in_progress, done, blocked) and completion percentage.

Requirements do NOT have a status field.  A requirement is satisfied when \
all steps that realize it are done.  This is a deliberate decision to \
separate intent from progress tracking.

A "Capability" layer (between Requirement and Step) was considered and \
deferred.  Steps currently serve as both architectural building blocks \
and work items.  If we later need separate dependency graphs for concepts \
vs tasks, Capability earns its place.

## Two operating modes

1. **Planning Partner** — this system IS the primary tool.  The validated \
   spec is the deliverable (architecture design, strategy, curriculum).
2. **Context Provider** — this system supports a primary tool \
   (e.g. Claude Code via MCP) by maintaining structured project state \
   that the primary tool consumes.

Both modes use the same spec, validation, and conversation loop.

## Constraints

- The LLM never sees internal graph structure (node IDs, edge types).
- A spec with only goals and requirements is valid — steps are optional.
- All mutations go through typed tool calls, not free-text prose.

## Integrity rules

- Every goal must have at least one requirement (SATISFIED_BY).
- Every requirement must have at least one step (REALIZED_BY).

When rules are violated the system produces Findings with severity, \
subject, message, and evidence.

## Your job

1. Review findings from the validation pass.
2. Explain issues clearly and suggest concrete actions.
3. Prioritize by severity (high → medium → low).
4. When proposing new goals, requirements, or steps — use the existing \
   naming conventions and link references correctly.
5. You do NOT modify the spec directly.  You advise on what changes \
   would resolve violations or advance the project.
"""


# ── Project-specific pre-flight quiz ──────────────────────────────────

META_QUIZ: list[FactualQuiz] = [
    FactualQuiz(
        id="meta-quiz-model",
        name="Data Model Quiz",
        question=(
            "What are the three layers of the project data model, and which one carries status?"
        ),
        expected_answer="goal, requirement, step, status",
        weight=1.5,
        min_score=0.5,
    ),
    FactualQuiz(
        id="meta-quiz-req-status",
        name="Requirement Status Quiz",
        question=(
            "Do requirements have a status field?  How do you determine "
            "whether a requirement is satisfied?"
        ),
        expected_answer="no, derived, step, done",
        weight=1.5,
        min_score=0.5,
    ),
    FactualQuiz(
        id="meta-quiz-modes",
        name="Operating Modes Quiz",
        question=(
            "What are the two operating modes of this system? "
            "Which one does not require external tool integration?"
        ),
        expected_answer="planning partner, context provider",
        weight=1.0,
        min_score=0.5,
    ),
    FactualQuiz(
        id="meta-quiz-constraint",
        name="LLM Constraint Quiz",
        question=("What does the LLM see — the KnowledgeGraph or the ProjectSpecification?  Why?"),
        expected_answer="ProjectSpecification, no graph, no IDs",
        weight=1.0,
        min_score=0.5,
    ),
    FactualQuiz(
        id="meta-quiz-mutation",
        name="Mutation Boundary Quiz",
        question="How does the LLM modify project state?",
        expected_answer="tool, typed, not prose",
        weight=1.0,
        min_score=0.5,
    ),
]


# ── Project-specific integrity rules ─────────────────────────────────


def meta_rules() -> list[IntegrityRule]:
    """Integrity rules for the dogfood project."""
    return [
        IntegrityRule(
            id="meta-rule-goal-req",
            name="Goal → Requirement",
            description="Every goal must have at least one requirement",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        ),
        IntegrityRule(
            id="meta-rule-req-step",
            name="Requirement → Step",
            description="Every requirement must have at least one step",
            applies_to_node_type="requirement",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["step"],
            minimum_count=1,
            severity="medium",
            failure_message_template="Requirement '{subject_name}' has no steps.",
        ),
    ]


# ── Full DomainConfig for the dogfood project ────────────────────────


def conversation_engine_meta_config(
    *,
    include_steps: bool = True,
) -> DomainConfig:
    """
    Complete DomainConfig for this repository as a project under management.

    Bundles the ProjectSpecification with project-specific rules, quiz,
    and system prompt.  This is what would be persisted and reloaded at
    the start of each session.
    """
    return DomainConfig(
        project_name="langgraph-ogar",
        project_spec=conversation_engine_meta_spec(include_steps=include_steps),
        rules=meta_rules(),
        quiz=list(META_QUIZ),
        system_prompt=META_SYSTEM_PROMPT,
        metadata={
            "dogfood": True,
            "modes": ["planning_partner", "context_provider"],
        },
    )


def infer_meta_spec_from_paths(
    paths: Iterable[str],
    *,
    project_name: str = "langgraph-ogar",
    include_steps: bool = False,
) -> ProjectSpecification:
    """
    Infer a lightweight goals/requirements spec from repository file paths.

    Heuristic by design — should be reviewed by a human before becoming
    durable project memory.
    """
    goals: dict[str, GoalSpec] = {}
    requirements: dict[str, RequirementSpec] = {}
    steps: list[StepSpec] = []

    def add_goal(name: str, statement: str) -> None:
        if name not in goals:
            goals[name] = GoalSpec(name=name, statement=statement)

    def add_requirement(name: str, goal_ref: str, description: str) -> None:
        if name not in requirements:
            requirements[name] = RequirementSpec(
                name=name,
                goal_ref=goal_ref,
                requirement_type="functional",
                description=description,
            )

    normalized = [p.replace("\\", "/") for p in paths]
    for path in normalized:
        if include_steps:
            steps.append(
                StepSpec(
                    name=path,
                    requirement_refs=[],
                    dependency_refs=[],
                    has_no_dependencies=True,
                    description="Observed repo file used as analysis evidence.",
                )
            )

        if "graph/" in path:
            add_goal(
                "Persistent Project Context",
                "Provide modular orchestration components that can be recomposed.",
            )
            add_requirement(
                "Explicit Workflow Control",
                "Persistent Project Context",
                "Represent orchestration as explicit graph topology and routing.",
            )

        if "tool_client/" in path:
            add_goal(
                "Constrained State Format",
                "Maintain trustworthy context and mutation boundaries over time.",
            )
            add_requirement(
                "Typed Tool Boundary",
                "Constrained State Format",
                "All tool interactions use typed schemas with validation.",
            )

        if "storage/" in path or "project_store" in path:
            add_goal(
                "Constrained State Format",
                "Maintain trustworthy context and mutation boundaries over time.",
            )
            add_requirement(
                "Durable Project State",
                "Constrained State Format",
                "Persist and reload project specification and control metadata.",
            )

        if "examples/" in path:
            add_goal(
                "Progressive Demo Platform",
                "Provide modular orchestration components that can be recomposed.",
            )
            add_requirement(
                "Demonstrable End-to-End Flow",
                "Progressive Demo Platform",
                "Include runnable examples for customer-facing demos.",
            )

    return ProjectSpecification(
        project_name=project_name,
        goals=list(goals.values()),
        requirements=list(requirements.values()),
        steps=steps if include_steps else [],
    )


def load_project_spec_fixture_json(path: str | Path) -> ProjectSpecification:
    """Load a JSON fixture into a typed ProjectSpecification."""
    raw = json.loads(Path(path).read_text())
    return ProjectSpecification.model_validate(raw)
