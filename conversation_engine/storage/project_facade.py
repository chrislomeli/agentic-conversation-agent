"""
ProjectSpecification ↔ KnowledgeGraph facade.

Converts the flat, business-level ``ProjectSpecification`` to/from the internal
``KnowledgeGraph`` representation.  This is the only place that knows about
node IDs, edge types, and edge directions.

The LLM never calls this directly — it is used inside the knowledge-graph
tool handler and by the ``ProjectStore`` pipeline.

Edge wiring rules (source --EDGE_TYPE--> target):
    Goal        --SATISFIED_BY-->  Requirement
    Requirement --REALIZED_BY-->   Step
    Step        --DEPENDS_ON-->    Dependency
    Constraint  --CONSTRAINS-->    (any node, not wired automatically)
"""

from __future__ import annotations

import json
import uuid

from conversation_engine.models.base import BaseEdge, EdgeType, NodeType
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.nodes import (
    Constraint,
    Dependency,
    Goal,
    Project,
    Requirement,
    Step,
)
from conversation_engine.models.project_spec import (
    ConstraintSpec,
    DependencySpec,
    GoalSpec,
    ProjectSpecification,  # backwards-compatible alias
    RequirementSpec,
    StepSpec,
)
from conversation_engine.storage.graph import KnowledgeGraph


# todo deprecate this
def _create_id(prefix: str) -> str:
    """Deterministic ID from a prefix and a human-readable name."""
    # slug = name.strip().lower().replace(" ", "-")
    return prefix + str(uuid.uuid4())


#
#
# # ── Snapshot → KnowledgeGraph ──────────────────────────────────────
#
# class SnapshotConversionError(Exception):
#     """Raised when a snapshot contains invalid references."""


class SnapshotConversionError(Exception):
    """Raised when a snapshot contains invalid references."""


def _get_quiz_edge_type(quiz_type: str) -> EdgeType:
    return "HAS_REASONING_QUIZ" if quiz_type.lower() == "reasoning" else "HAS_FACTUAL_QUIZ"


def project_to_graph(project: DomainConfig) -> KnowledgeGraph:
    graph = KnowledgeGraph()

    # 1. Add the root Project node
    project_id = _create_id("project")
    project_node = Project(
        id=project_id,  # _slugify("project", project.project_name),
        name=project.project_name,
        system_prompt=project.system_prompt or "",
        metadata=json.dumps(project.metadata) or "",
    )
    graph.add_node(project_node)

    # ── Name → ID registries (built as nodes are added) ────────────
    goal_ids: dict[str, str] = {}
    req_ids: dict[str, str] = {}
    step_ids: dict[str, str] = {}
    dep_ids: dict[str, str] = {}

    # ── rules ──────────────────────────────────────────────────────
    for spec in project.rules or []:
        nid = _create_id("rule")
        # Create a copy with the new ID
        rule_with_id = spec.model_copy(update={"id": nid})
        graph.add_node(rule_with_id)
        graph.add_edge(
            BaseEdge(
                edge_type="HAS_RULE",
                source_id=project_id,
                target_id=nid,
            )
        )

    # ── Quiz ──────────────────────────────────────────────────────
    for spec in project.quiz or []:
        quiz_type = spec.quiz_type
        if quiz_type.startswith("reason"):
            pass
        nid = _create_id(quiz_type)
        # Create a copy with the new ID
        quiz_with_id = spec.model_copy(update={"id": nid})
        graph.add_node(quiz_with_id)
        graph.add_edge(
            BaseEdge(
                edge_type=_get_quiz_edge_type(quiz_type),
                source_id=project_id,
                target_id=nid,
            )
        )

    if not (project_spec := project.project_spec):
        pass
    # ── Project Specification ──────────────────────────────────────────────────────
    else:
        # ── Goals ──────────────────────────────────────────────────────
        for spec in project_spec.goals or []:
            nid = _create_id("goal")
            goal_ids[spec.name] = nid
            graph.add_node(Goal(id=nid, name=spec.name, statement=spec.statement))
            # Wire: wire to project
            graph.add_edge(
                BaseEdge(
                    edge_type="HAS_GOAL",
                    source_id=project_id,
                    target_id=nid,
                )
            )

        # ── Requirements (→ Goal via SATISFIED_BY) ─────────────────────
        for spec in project_spec.requirements or []:
            nid = _create_id("req")
            req_ids[spec.name] = nid
            graph.add_node(
                Requirement(
                    id=nid,
                    name=spec.name,
                    requirement_type=spec.requirement_type,
                    description=spec.description,
                )
            )
            # Wire: wire to project
            graph.add_edge(
                BaseEdge(
                    edge_type="HAS_REQUIREMENT",
                    source_id=project_id,
                    target_id=nid,
                )
            )

            # Wire: Goal --SATISFIED_BY--> Requirement
            if spec.goal_ref:
                goal_id = goal_ids.get(spec.goal_ref)
                if goal_id is None:
                    raise SnapshotConversionError(
                        f"Requirement '{spec.name}' references unknown goal '{spec.goal_ref}'. "
                        f"Known goals: {sorted(goal_ids)}"
                    )
                graph.add_edge(
                    BaseEdge(
                        edge_type="SATISFIED_BY",
                        source_id=goal_id,
                        target_id=nid,
                    )
                )

        # ── Dependencies (added before steps so refs resolve) ─────
        for spec in project_spec.dependencies or []:
            nid = _create_id("dep")
            dep_ids[spec.name] = nid
            graph.add_node(
                Dependency(
                    id=nid,
                    name=spec.name,
                    description=spec.description,
                )
            )
            # Wire: wire to project  todo - wire dependencies to a goal instead of at the project level?
            graph.add_edge(
                BaseEdge(
                    edge_type="HAS_DEPENDENCY",
                    source_id=project_id,
                    target_id=nid,
                )
            )

        # ── Steps (→ Requirement via REALIZED_BY, → Dependency via DEPENDS_ON)
        for spec in project_spec.steps or []:
            nid = _create_id("step")
            step_ids[spec.name] = nid
            graph.add_node(
                Step(
                    id=nid,
                    name=spec.name,
                    description=spec.description,
                    has_no_dependencies=spec.has_no_dependencies,
                )
            )
            # Wire: wire to project  todo - wire steps to a goal instead of at the project level?
            graph.add_edge(
                BaseEdge(
                    edge_type="HAS_STEP",
                    source_id=project_id,
                    target_id=nid,
                )
            )

            for ref in spec.requirement_refs:
                req_id = req_ids.get(ref)
                if req_id is None:
                    raise SnapshotConversionError(
                        f"Step '{spec.name}' references unknown requirement '{ref}'. "
                        f"Known requirements: {sorted(req_ids)}"
                    )
                graph.add_edge(
                    BaseEdge(
                        edge_type="REALIZED_BY",
                        source_id=req_id,
                        target_id=nid,
                    )
                )
            for ref in spec.dependency_refs:
                dep_id = dep_ids.get(ref)
                if dep_id is None:
                    raise SnapshotConversionError(
                        f"Step '{spec.name}' references unknown dependency '{ref}'. "
                        f"Known dependencies: {sorted(dep_ids)}"
                    )
                graph.add_edge(
                    BaseEdge(
                        edge_type="DEPENDS_ON",
                        source_id=nid,
                        target_id=dep_id,
                    )
                )

        # ── Constraints (standalone — no automatic edges) ──────────────
        for spec in project_spec.constraints or []:
            nid = _create_id("cstr")
            graph.add_node(Constraint(id=nid, name=spec.name, statement=spec.statement))
            # Wire: wire to project  todo - wire steps to a goal instead of at the project level?
            graph.add_edge(
                BaseEdge(
                    edge_type="HAS_CONSTRAINT",
                    source_id=project_id,
                    target_id=nid,
                )
            )

    return graph


# ── KnowledgeGraph → Snapshot ──────────────────────────────────────


def graph_to_domain_config(graph: KnowledgeGraph) -> DomainConfig:
    """
    Convert a ``KnowledgeGraph`` back to a ``DomainConfig``.

    Reconstructs the complete domain configuration including:
    - Project metadata (from root Project node)
    - ProjectSpecification (existing logic)
    - Quiz nodes (both FactualQuiz and ReasoningQuiz)
    - Rule nodes
    """
    # ── Find the root Project node ──────────────────────────────────────
    project_nodes = graph.get_nodes_by_type(NodeType.PROJECT)
    if not project_nodes:
        raise ValueError("No project node found in graph")
    project_node = project_nodes[0]
    system_prompt = project_node.system_prompt
    meta_data = project_node.metadata
    project_name = project_node.name

    # ── Extract quiz nodes ──────────────────────────────────────────────
    quiz_nodes: list[ValidationQuiz] = graph.get_nodes_by_type(NodeType.QUIZ)  # type: ignore

    # ── Extract rule nodes ──────────────────────────────────────────────
    rule_nodes: list[IntegrityRule] = graph.get_nodes_by_type(NodeType.RULE)  # type: ignore

    # ── Get ProjectSpecification using existing logic ───────────────────
    project_specification = graph_to_snapshot(project_node.name, graph)

    # ── Parse metadata from JSON string ─────────────────────────────────
    metadata = {}
    if meta_data:
        try:
            metadata = json.loads(meta_data)
        except json.JSONDecodeError:
            # If metadata is not valid JSON, treat as empty dict
            metadata = {}

    # ── Build complete DomainConfig ─────────────────────────────────────
    return DomainConfig(
        project_name=project_name,
        project_spec=project_specification,
        quiz=quiz_nodes,
        rules=rule_nodes,
        system_prompt=system_prompt or "",
        metadata=metadata,
    )


def graph_to_snapshot(project_name: str, graph: KnowledgeGraph) -> ProjectSpecification:
    """
    Convert a ``KnowledgeGraph`` back to a ``ProjectSpecification``.

    Reconstructs name-based references by following edges in reverse.
    Nodes without matching types are silently skipped (forward-compatible
    with future node types the snapshot doesn't yet model).
    """
    # ── Collect nodes by type ──────────────────────────────────────
    goals_by_id: dict[str, Goal] = {}
    reqs_by_id: dict[str, Requirement] = {}
    steps_by_id: dict[str, Step] = {}
    deps_by_id: dict[str, Dependency] = {}
    constraints_by_id: dict[str, Constraint] = {}

    for node in graph.get_all_nodes():
        if isinstance(node, Goal):
            goals_by_id[node.id] = node
        elif isinstance(node, Requirement):
            reqs_by_id[node.id] = node
        elif isinstance(node, Step):
            steps_by_id[node.id] = node
        elif isinstance(node, Dependency):
            deps_by_id[node.id] = node
        elif isinstance(node, Constraint):
            constraints_by_id[node.id] = node

    # ── Build reverse lookups from edges ───────────────────────────
    # Requirement → goal name  (Goal --SATISFIED_BY--> Req)
    req_to_goal: dict[str, str] = {}
    for edge in graph.get_edges_by_type("SATISFIED_BY"):
        if edge.target_id in reqs_by_id and edge.source_id in goals_by_id:
            req_to_goal[edge.target_id] = goals_by_id[edge.source_id].name

    # Step → requirement names  (Req --REALIZED_BY--> Step)
    step_to_reqs: dict[str, list[str]] = {}
    for edge in graph.get_edges_by_type("REALIZED_BY"):
        if edge.target_id in steps_by_id and edge.source_id in reqs_by_id:
            step_to_reqs.setdefault(edge.target_id, []).append(reqs_by_id[edge.source_id].name)

    # Step → dependency names  (Step --DEPENDS_ON--> Dep)
    step_to_deps: dict[str, list[str]] = {}
    for edge in graph.get_edges_by_type("DEPENDS_ON"):
        if edge.source_id in steps_by_id and edge.target_id in deps_by_id:
            step_to_deps.setdefault(edge.source_id, []).append(deps_by_id[edge.target_id].name)

    # ── Assemble specs ─────────────────────────────────────────────
    goal_specs = [GoalSpec(name=g.name, statement=g.statement) for g in goals_by_id.values()]

    req_specs = [
        RequirementSpec(
            name=r.name,
            goal_ref=req_to_goal.get(r.id, ""),
            requirement_type=r.requirement_type,
            description=r.description,
        )
        for r in reqs_by_id.values()
    ]

    step_specs = [
        StepSpec(
            name=s.name,
            requirement_refs=step_to_reqs.get(s.id, []),
            dependency_refs=step_to_deps.get(s.id, []),
            has_no_dependencies=s.has_no_dependencies,
            description=s.description,
        )
        for s in steps_by_id.values()
    ]

    constraint_specs = [
        ConstraintSpec(name=c.name, statement=c.statement) for c in constraints_by_id.values()
    ]

    dep_specs = [
        DependencySpec(name=d.name, description=d.description) for d in deps_by_id.values()
    ]

    return ProjectSpecification(
        project_name=project_name,
        goals=goal_specs,
        requirements=req_specs,
        steps=step_specs,
        constraints=constraint_specs,
        dependencies=dep_specs,
    )
