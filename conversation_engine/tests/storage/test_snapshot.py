"""
Tests for ProjectSnapshot, snapshot facade, and project_spec tool.

Covers:
- ProjectSnapshot model construction and validation
- snapshot_to_graph() — forward conversion with correct nodes, edges, IDs
- graph_to_snapshot() — reverse conversion preserving names and refs
- Round-trip: snapshot → graph → snapshot
- SnapshotConversionError on bad references
- make_project_spec_tool() — CREATE, READ, UPDATE, DELETE via ToolSpec
"""

from __future__ import annotations

import pytest

from conversation_engine.infrastructure.tool_client.project_graph_tools import (
    ProjectGraphInput,
    make_project_spec_tool,
)
from conversation_engine.models.base import BaseEdge
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.nodes import (
    Constraint,
    Dependency,
    Goal,
    Requirement,
    Step,
)
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.project_store import InMemoryProjectStore
from conversation_engine.storage.snapshot import (
    ConstraintSpec,
    DependencySpec,
    GoalSpec,
    ProjectSnapshot,
    RequirementSpec,
    StepSpec,
)
from conversation_engine.storage.snapshot_facade import (
    SnapshotConversionError,
    _slugify,
    graph_to_snapshot,
    snapshot_to_graph,
)

# ── Helpers ────────────────────────────────────────────────────────


def _sample_snapshot() -> ProjectSnapshot:
    """A realistic snapshot with all five entity types."""
    return ProjectSnapshot(
        project_name="acme",
        goals=[
            GoalSpec(name="User Authentication", statement="Users can log in securely"),
        ],
        requirements=[
            RequirementSpec(
                name="OAuth Support",
                goal_ref="User Authentication",
                requirement_type="functional",
                description="Support OAuth 2.0 providers",
            ),
        ],
        steps=[
            StepSpec(
                name="Auth Service",
                requirement_refs=["OAuth Support"],
                dependency_refs=["Redis"],
                description="Handles authentication flows",
            ),
        ],
        constraints=[
            ConstraintSpec(name="GDPR", statement="Must comply with GDPR"),
        ],
        dependencies=[
            DependencySpec(name="Redis", description="Session cache"),
        ],
    )


# ── ProjectSnapshot model ─────────────────────────────────────────


class TestProjectSnapshot:
    def test_minimal_snapshot(self):
        snap = ProjectSnapshot(project_name="empty")
        assert snap.project_name == "empty"
        assert snap.goals == []
        assert snap.requirements == []

    def test_full_snapshot(self):
        snap = _sample_snapshot()
        assert snap.project_name == "acme"
        assert len(snap.goals) == 1
        assert len(snap.requirements) == 1
        assert len(snap.steps) == 1
        assert len(snap.constraints) == 1
        assert len(snap.dependencies) == 1

    def test_snapshot_serialises_to_dict(self):
        snap = _sample_snapshot()
        d = snap.model_dump()
        assert d["project_name"] == "acme"
        assert d["goals"][0]["name"] == "User Authentication"

    def test_snapshot_round_trip_json(self):
        snap = _sample_snapshot()
        json_str = snap.model_dump_json()
        restored = ProjectSnapshot.model_validate_json(json_str)
        assert restored == snap


# ── Slugify helper ─────────────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert _slugify("goal", "User Authentication") == "goal-user-authentication"

    def test_strips_whitespace(self):
        assert _slugify("req", "  foo bar  ") == "req-foo-bar"


# ── snapshot_to_graph ──────────────────────────────────────────────


class TestSnapshotToGraph:
    def test_creates_all_nodes(self):
        graph = snapshot_to_graph(_sample_snapshot())
        assert graph.node_count() == 5  # goal, req, step, cstr, dep

    def test_creates_correct_edges(self):
        graph = snapshot_to_graph(_sample_snapshot())
        # Goal --SATISFIED_BY--> Requirement
        # Requirement --REALIZED_BY--> Step
        # Step --DEPENDS_ON--> Dependency
        assert graph.edge_count() == 3

    def test_goal_node(self):
        graph = snapshot_to_graph(_sample_snapshot())
        node = graph.get_node("goal-user-authentication")
        assert node is not None
        assert isinstance(node, Goal)
        assert node.name == "User Authentication"
        assert node.statement == "Users can log in securely"

    def test_requirement_node(self):
        graph = snapshot_to_graph(_sample_snapshot())
        node = graph.get_node("req-oauth-support")
        assert node is not None
        assert isinstance(node, Requirement)
        assert node.requirement_type == "functional"

    def test_satisfied_by_edge(self):
        graph = snapshot_to_graph(_sample_snapshot())
        edge = graph.get_edge("goal-user-authentication", "SATISFIED_BY", "req-oauth-support")
        assert edge is not None

    def test_realized_by_req_to_comp(self):
        graph = snapshot_to_graph(_sample_snapshot())
        edge = graph.get_edge("req-oauth-support", "REALIZED_BY", "step-auth-service")
        assert edge is not None

    def test_depends_on_edge(self):
        graph = snapshot_to_graph(_sample_snapshot())
        edge = graph.get_edge("step-auth-service", "DEPENDS_ON", "dep-redis")
        assert edge is not None

    def test_constraint_node(self):
        graph = snapshot_to_graph(_sample_snapshot())
        node = graph.get_node("cstr-gdpr")
        assert node is not None
        assert isinstance(node, Constraint)
        assert node.statement == "Must comply with GDPR"

    def test_step_has_no_dependencies_flag(self):
        snap = ProjectSnapshot(
            project_name="test",
            steps=[
                StepSpec(name="Standalone", has_no_dependencies=True),
            ],
        )
        graph = snapshot_to_graph(snap)
        node = graph.get_node_typed("step-standalone", Step)
        assert node is not None
        assert node.has_no_dependencies is True

    def test_empty_snapshot_produces_empty_graph(self):
        graph = snapshot_to_graph(ProjectSnapshot(project_name="empty"))
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_bad_goal_ref_raises(self):
        snap = ProjectSnapshot(
            project_name="test",
            requirements=[
                RequirementSpec(name="R1", goal_ref="NonexistentGoal"),
            ],
        )
        with pytest.raises(SnapshotConversionError, match="unknown goal"):
            snapshot_to_graph(snap)

    def test_bad_requirement_ref_on_step_raises(self):
        snap = ProjectSnapshot(
            project_name="test",
            steps=[
                StepSpec(name="X", requirement_refs=["Nonexistent"]),
            ],
        )
        with pytest.raises(SnapshotConversionError, match="unknown requirement"):
            snapshot_to_graph(snap)

    def test_bad_dependency_ref_raises(self):
        snap = ProjectSnapshot(
            project_name="test",
            steps=[
                StepSpec(name="X", dependency_refs=["Nonexistent"]),
            ],
        )
        with pytest.raises(SnapshotConversionError, match="unknown dependency"):
            snapshot_to_graph(snap)

    def test_multiple_requirements_per_goal(self):
        snap = ProjectSnapshot(
            project_name="test",
            goals=[GoalSpec(name="G", statement="g")],
            requirements=[
                RequirementSpec(name="R1", goal_ref="G"),
                RequirementSpec(name="R2", goal_ref="G"),
            ],
        )
        graph = snapshot_to_graph(snap)
        edges = graph.get_outgoing_edges("goal-g", "SATISFIED_BY")
        assert len(edges) == 2

    def test_step_with_multiple_requirement_refs(self):
        snap = ProjectSnapshot(
            project_name="test",
            goals=[GoalSpec(name="G", statement="g")],
            requirements=[
                RequirementSpec(name="R1", goal_ref="G"),
                RequirementSpec(name="R2", goal_ref="G"),
            ],
            steps=[
                StepSpec(name="S", requirement_refs=["R1", "R2"]),
            ],
        )
        graph = snapshot_to_graph(snap)
        # R1 --REALIZED_BY--> S  and  R2 --REALIZED_BY--> S
        assert graph.get_edge("req-r1", "REALIZED_BY", "step-s") is not None
        assert graph.get_edge("req-r2", "REALIZED_BY", "step-s") is not None


# ── graph_to_snapshot ──────────────────────────────────────────────


class TestGraphToSnapshot:
    def test_basic_reverse(self):
        """Build a graph manually and convert back to snapshot."""
        graph = KnowledgeGraph()
        graph.add_node(Goal(id="g1", name="G1", statement="Goal one"))
        graph.add_node(Requirement(id="r1", name="R1", requirement_type="functional"))
        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="g1", target_id="r1"))

        snap = graph_to_snapshot("test", graph)
        assert snap.project_name == "test"
        assert len(snap.goals) == 1
        assert snap.goals[0].name == "G1"
        assert len(snap.requirements) == 1
        assert snap.requirements[0].goal_ref == "G1"

    def test_empty_graph(self):
        snap = graph_to_snapshot("empty", KnowledgeGraph())
        assert snap.goals == []
        assert snap.requirements == []

    def test_step_depends_on(self):
        graph = KnowledgeGraph()
        graph.add_node(Step(id="s1", name="Step1"))
        graph.add_node(Dependency(id="d1", name="Dep1"))
        graph.add_edge(BaseEdge(edge_type="DEPENDS_ON", source_id="s1", target_id="d1"))

        snap = graph_to_snapshot("test", graph)
        assert len(snap.steps) == 1
        assert snap.steps[0].dependency_refs == ["Dep1"]

    def test_unknown_node_types_silently_skipped(self):
        """Nodes the snapshot doesn't model (e.g. Feature) are skipped."""
        from conversation_engine.models.nodes import Feature

        graph = KnowledgeGraph()
        graph.add_node(Feature(id="f1", name="F1", description="A feature"))
        graph.add_node(Goal(id="g1", name="G1", statement="s"))

        snap = graph_to_snapshot("test", graph)
        assert len(snap.goals) == 1
        # Feature is silently skipped — no error


# ── Round-trip ─────────────────────────────────────────────────────


class TestRoundTrip:
    def test_snapshot_to_graph_and_back(self):
        """snapshot → graph → snapshot preserves all business data."""
        original = _sample_snapshot()
        graph = snapshot_to_graph(original)
        restored = graph_to_snapshot("acme", graph)

        assert restored.project_name == original.project_name
        assert len(restored.goals) == len(original.goals)
        assert len(restored.requirements) == len(original.requirements)
        assert len(restored.steps) == len(original.steps)
        assert len(restored.constraints) == len(original.constraints)
        assert len(restored.dependencies) == len(original.dependencies)

        # Check content
        assert restored.goals[0].name == "User Authentication"
        assert restored.requirements[0].goal_ref == "User Authentication"
        assert restored.steps[0].requirement_refs == ["OAuth Support"]
        assert restored.steps[0].dependency_refs == ["Redis"]

    def test_empty_round_trip(self):
        original = ProjectSnapshot(project_name="empty")
        graph = snapshot_to_graph(original)
        restored = graph_to_snapshot("empty", graph)
        assert restored.goals == []
        assert restored.requirements == []


# ── make_project_spec_tool ─────────────────────────────────────────


class TestProjectSpecTool:
    def _make_tool(self):
        store = InMemoryProjectStore()
        spec = make_project_spec_tool(store)
        return spec, store

    def test_tool_spec_metadata(self):
        spec, _ = self._make_tool()
        assert spec.name == "project_spec"
        assert "CREATE" in spec.description
        assert "READ" in spec.description
        assert "UPDATE" in spec.description
        assert "DELETE" in spec.description

    def test_create_success(self):
        spec, store = self._make_tool()
        inp = ProjectGraphInput(method="CREATE", payload=_sample_snapshot())
        out = spec.handler(inp)
        assert out.success is True
        assert "created" in out.message.lower()
        assert store.exists("acme")

    def test_create_no_payload(self):
        spec, _ = self._make_tool()
        inp = ProjectGraphInput(method="CREATE")
        out = spec.handler(inp)
        assert out.success is False
        assert "payload" in out.message.lower()

    def test_create_duplicate_fails(self):
        spec, _ = self._make_tool()
        inp = ProjectGraphInput(method="CREATE", payload=_sample_snapshot())
        spec.handler(inp)
        out = spec.handler(inp)
        assert out.success is False
        assert "already exists" in out.message.lower()

    def test_create_bad_ref_fails(self):
        spec, _ = self._make_tool()
        bad_snap = ProjectSnapshot(
            project_name="bad",
            requirements=[RequirementSpec(name="R", goal_ref="NoGoal")],
        )
        inp = ProjectGraphInput(method="CREATE", payload=bad_snap)
        out = spec.handler(inp)
        assert out.success is False
        assert "invalid snapshot" in out.message.lower()

    def test_read_success(self):
        spec, _ = self._make_tool()
        spec.handler(ProjectGraphInput(method="CREATE", payload=_sample_snapshot()))
        inp = ProjectGraphInput(method="READ", key="acme")
        out = spec.handler(inp)
        assert out.success is True
        assert out.payload is not None
        assert out.payload.project_name == "acme"
        assert len(out.payload.goals) == 1

    def test_read_not_found(self):
        spec, _ = self._make_tool()
        inp = ProjectGraphInput(method="READ", key="nope")
        out = spec.handler(inp)
        assert out.success is False
        assert "not found" in out.message.lower()

    def test_read_no_key(self):
        spec, _ = self._make_tool()
        inp = ProjectGraphInput(method="READ")
        out = spec.handler(inp)
        assert out.success is False
        assert "key" in out.message.lower()

    def test_delete_success(self):
        spec, store = self._make_tool()
        spec.handler(ProjectGraphInput(method="CREATE", payload=_sample_snapshot()))
        assert store.exists("acme")
        inp = ProjectGraphInput(method="DELETE", key="acme")
        out = spec.handler(inp)
        assert out.success is True
        assert not store.exists("acme")

    def test_delete_not_found(self):
        spec, _ = self._make_tool()
        inp = ProjectGraphInput(method="DELETE", key="nope")
        out = spec.handler(inp)
        assert out.success is False

    def test_delete_no_key(self):
        spec, _ = self._make_tool()
        inp = ProjectGraphInput(method="DELETE")
        out = spec.handler(inp)
        assert out.success is False

    def test_update_success(self):
        """UPDATE full-replaces the spec and returns the new payload."""
        spec, store = self._make_tool()
        spec.handler(ProjectGraphInput(method="CREATE", payload=_sample_snapshot()))

        updated = _sample_snapshot()
        updated.goals.append(GoalSpec(name="New Goal", statement="Added later"))
        updated.requirements.append(RequirementSpec(name="New Req", goal_ref="New Goal"))
        out = spec.handler(ProjectGraphInput(method="UPDATE", key="acme", payload=updated))
        assert out.success is True
        assert "updated" in out.message.lower()
        assert out.payload is not None
        assert len(out.payload.goals) == 2

        # Verify via READ
        read_out = spec.handler(ProjectGraphInput(method="READ", key="acme"))
        assert read_out.payload is not None
        assert len(read_out.payload.goals) == 2

    def test_update_preserves_control_fields(self):
        """UPDATE replaces spec but preserves rules, quiz, system_prompt, etc."""
        tool, store = self._make_tool()

        # Seed with a full DomainConfig that has control fields
        rule = IntegrityRule(
            id="r1",
            name="test rule",
            description="d",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            edge_type="SATISFIED_BY",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )
        config = DomainConfig(
            project_name="acme",
            project_spec=_sample_snapshot(),
            rules=[rule],
            system_prompt="Do not hallucinate.",
        )
        store.save(config)

        # UPDATE spec only
        new_spec = ProjectSnapshot(
            project_name="acme",
            goals=[GoalSpec(name="Only Goal", statement="Replaced")],
        )
        out = tool.handler(ProjectGraphInput(method="UPDATE", key="acme", payload=new_spec))
        assert out.success is True

        # Verify control fields survived
        loaded = store.load("acme")
        assert loaded is not None
        assert len(loaded.project_spec.goals) == 1
        assert loaded.project_spec.goals[0].name == "Only Goal"
        assert loaded.rules is not None
        assert len(loaded.rules) == 1
        assert loaded.rules[0].id == "r1"
        assert loaded.system_prompt == "Do not hallucinate."

    def test_update_not_found(self):
        """UPDATE on nonexistent project fails."""
        spec, _ = self._make_tool()
        out = spec.handler(
            ProjectGraphInput(
                method="UPDATE",
                key="nope",
                payload=_sample_snapshot(),
            )
        )
        assert out.success is False
        assert "not found" in out.message.lower()

    def test_update_no_key(self):
        """UPDATE without key fails."""
        spec, _ = self._make_tool()
        out = spec.handler(ProjectGraphInput(method="UPDATE", payload=_sample_snapshot()))
        assert out.success is False
        assert "key" in out.message.lower()

    def test_update_no_payload(self):
        """UPDATE without payload fails."""
        spec, _ = self._make_tool()
        out = spec.handler(ProjectGraphInput(method="UPDATE", key="acme"))
        assert out.success is False
        assert "payload" in out.message.lower()

    def test_update_bad_ref_fails(self):
        """UPDATE with invalid refs fails."""
        spec, store = self._make_tool()
        spec.handler(ProjectGraphInput(method="CREATE", payload=_sample_snapshot()))
        bad = ProjectSnapshot(
            project_name="acme",
            requirements=[RequirementSpec(name="R", goal_ref="NoGoal")],
        )
        out = spec.handler(ProjectGraphInput(method="UPDATE", key="acme", payload=bad))
        assert out.success is False
        assert "invalid snapshot" in out.message.lower()

    def test_create_read_round_trip_preserves_data(self):
        """CREATE then READ returns equivalent data."""
        spec, _ = self._make_tool()
        original = _sample_snapshot()
        spec.handler(ProjectGraphInput(method="CREATE", payload=original))
        out = spec.handler(ProjectGraphInput(method="READ", key="acme"))

        assert out.payload is not None
        restored = out.payload
        assert restored.goals[0].name == original.goals[0].name
        assert restored.requirements[0].goal_ref == original.requirements[0].goal_ref
        assert restored.steps[0].requirement_refs == original.steps[0].requirement_refs
        assert restored.steps[0].dependency_refs == original.steps[0].dependency_refs

    def test_tool_works_with_local_tool_client(self):
        """Integration: tool works through LocalToolClient envelope."""
        from conversation_engine.infrastructure.tool_client import (
            LocalToolClient,
            ToolRegistry,
        )

        store = InMemoryProjectStore()
        spec = make_project_spec_tool(store)
        reg = ToolRegistry()
        reg.register(spec)
        client = LocalToolClient(reg)

        # CREATE
        env = client.call(
            "project_spec",
            {
                "method": "CREATE",
                "payload": _sample_snapshot().model_dump(),
            },
        )
        assert not env.is_error
        assert env.structured["success"] is True

        # READ
        env = client.call("project_spec", {"method": "READ", "key": "acme"})
        assert not env.is_error
        assert env.structured["success"] is True
        assert env.structured["payload"]["project_name"] == "acme"

        # UPDATE
        updated = _sample_snapshot()
        updated.goals.append(GoalSpec(name="New Goal", statement="Added"))
        env = client.call(
            "project_spec",
            {
                "method": "UPDATE",
                "key": "acme",
                "payload": updated.model_dump(),
            },
        )
        assert not env.is_error
        assert env.structured["success"] is True

        # Verify UPDATE via READ
        env = client.call("project_spec", {"method": "READ", "key": "acme"})
        assert len(env.structured["payload"]["goals"]) == 2
