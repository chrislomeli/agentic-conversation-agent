"""
Tests for serialization (to_dict / from_dict) and FileProjectStore.

Coverage:
- KnowledgeGraph: round-trip, empty graph, polymorphic nodes, edges
- DomainConfig: round-trip with all fields, minimal config, None fields preserved
- FileProjectStore: save/load/delete/list/exists on disk
- End-to-end: DomainConfig → JSON file → DomainConfig → ArchitecturalOntologyContext
"""

from __future__ import annotations

import json
import pytest

from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.project_spec import ProjectSpecification
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.file_project_store import FileProjectStore
from conversation_engine.storage.snapshot_facade import graph_to_snapshot
from conversation_engine.models.rule_node import IntegrityRule

# from conversation_engine.models.query_node import (
#     GraphQueryPattern,
#     EdgeCheck,
# )
from conversation_engine.models.validation_quiz import FactualQuiz
from conversation_engine.graph.architectural_context import (
    ArchitecturalOntologyContext,
)
from conversation_engine.models import (
    Goal,
    Requirement,
    Step,
)
from conversation_engine.models.base import BaseEdge


# ── Helpers ──────────────────────────────────────────────────────────


def _sample_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(Goal(id="g1", name="Goal 1", statement="A goal"))
    g.add_node(Requirement(id="r1", name="Req 1"))
    g.add_node(Step(id="step1", name="Step 1", has_no_dependencies=True))
    g.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="g1", target_id="r1"))
    g.add_edge(BaseEdge(edge_type="REALIZED_BY", source_id="r1", target_id="step1"))
    return g


def _sample_rules() -> list[IntegrityRule]:
    return [
        IntegrityRule(
            id="rule-goal-req",
            name="Goal → Requirement",
            description="Every goal must have at least one requirement",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            # edge_type="SATISFIED_BY",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        ),
    ]


def _sample_quiz() -> list[FactualQuiz]:
    return [
        FactualQuiz(
            id="test-quiz-node-types",
            name="Test Node Types Quiz",
            question="What node types exist?",
            expected_answer="goal, requirement",
            weight=2.0,
            min_score=0.6,
        ),
    ]


# def _sample_query_patterns() -> list[GraphQueryPattern]:
#     return [
#         GraphQueryPattern(
#             id="qp-1",
#             name="Goal gap check",
#             description="Find goals without requirements",
#             subject_node_type="goal",
#             query_intent="gap_detection",
#             checks=[
#                 EdgeCheck(
#                     edge_type="SATISFIED_BY",
#                     target_node_types=["requirement"],
#                     expected_min_count=1,
#                 ),
#             ],
#             output_kind="finding_set",
#         ),
#     ]


def _sample_spec() -> ProjectSpecification:
    return graph_to_snapshot("test-project", _sample_graph())


def _full_config() -> DomainConfig:
    return DomainConfig(
        project_name="test-project",
        project_spec=_sample_spec(),
        rules=_sample_rules(),
        quiz=_sample_quiz(),
        # query_patterns=_sample_query_patterns(),
        system_prompt="You are a test assistant.",
        metadata={"version": "1.0", "owner": "test"},
    )


# ═════════════════════════════════════════════════════════════════════
#  KnowledgeGraph serialization
# ═════════════════════════════════════════════════════════════════════


class TestKnowledgeGraphSerialization:
    def test_round_trip(self):
        original = _sample_graph()
        data = original.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        assert restored.node_count() == original.node_count()
        assert restored.edge_count() == original.edge_count()

    def test_empty_graph(self):
        g = KnowledgeGraph()
        data = g.to_dict()
        assert data == {"nodes": [], "edges": []}
        restored = KnowledgeGraph.from_dict(data)
        assert restored.node_count() == 0
        assert restored.edge_count() == 0

    def test_preserves_node_types(self):
        original = _sample_graph()
        data = original.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        assert isinstance(restored.get_node("g1"), Goal)
        assert isinstance(restored.get_node("r1"), Requirement)
        assert isinstance(restored.get_node("step1"), Step)

    def test_preserves_node_fields(self):
        original = _sample_graph()
        data = original.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        goal = restored.get_node("g1")
        assert goal.name == "Goal 1"
        assert goal.statement == "A goal"
        step = restored.get_node("step1")
        assert step.has_no_dependencies is True

    def test_preserves_edges(self):
        original = _sample_graph()
        data = original.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        edge = restored.get_edge("g1", "SATISFIED_BY", "r1")
        assert edge is not None
        assert edge.source_id == "g1"
        assert edge.target_id == "r1"
        edge2 = restored.get_edge("r1", "REALIZED_BY", "step1")
        assert edge2 is not None
        assert edge2.source_id == "r1"
        assert edge2.target_id == "step1"

    def test_json_safe(self):
        data = _sample_graph().to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = KnowledgeGraph.from_dict(parsed)
        assert restored.node_count() == 3

    def test_unknown_node_type_raises(self):
        data = {"nodes": [{"_type": "alien", "id": "x", "name": "X"}], "edges": []}
        with pytest.raises(ValueError, match="is not a valid NodeType"):
            KnowledgeGraph.from_dict(data)

    def test_missing_type_raises(self):
        data = {"nodes": [{"id": "x", "name": "X"}], "edges": []}
        with pytest.raises(ValueError, match="missing '_type'"):
            KnowledgeGraph.from_dict(data)


# ═════════════════════════════════════════════════════════════════════
#  DomainConfig serialization
# ═════════════════════════════════════════════════════════════════════


class TestDomainConfigSerialization:
    def test_round_trip_full(self):
        original = _full_config()
        data = original.to_dict()
        restored = DomainConfig.from_dict(data)
        assert restored.project_name == "test-project"
        assert restored.project_spec is not None
        assert len(restored.project_spec.goals) == 1
        assert len(restored.project_spec.requirements) == 1
        assert len(restored.rules) == 1
        assert restored.rules[0].id == "rule-goal-req"
        assert len(restored.quiz) == 1
        assert restored.quiz[0].question == "What node types exist?"
        assert restored.quiz[0].expected_answer == "goal, requirement"
        assert restored.quiz[0].weight == 2.0
        assert restored.quiz[0].min_score == 0.6
        # assert len(restored.query_patterns) == 1
        # assert restored.query_patterns[0].id == "qp-1"
        assert restored.system_prompt == "You are a test assistant."
        assert restored.metadata == {"version": "1.0", "owner": "test"}

    def test_round_trip_minimal(self):
        original = DomainConfig(project_name="bare")
        data = original.to_dict()
        restored = DomainConfig.from_dict(data)
        assert restored.project_name == "bare"
        assert restored.project_spec is None
        assert restored.rules is None
        assert restored.quiz is None
        # assert restored.query_patterns is None
        assert restored.system_prompt is None
        assert restored.metadata == {}

    def test_json_safe(self):
        data = _full_config().to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = DomainConfig.from_dict(parsed)
        assert restored.project_name == "test-project"
        assert restored.project_spec is not None
        assert len(restored.project_spec.goals) == 1


# ═════════════════════════════════════════════════════════════════════
#  FileProjectStore
# ═════════════════════════════════════════════════════════════════════


class TestFileProjectStore:
    def test_save_and_load(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        store.save(_full_config())
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.project_name == "test-project"
        assert loaded.project_spec is not None
        assert len(loaded.project_spec.goals) == 1

    def test_load_nonexistent(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        assert store.load("nope") is None

    def test_upsert_replaces(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        store.save(_full_config())
        updated = DomainConfig(project_name="test-project", system_prompt="new prompt")
        store.save(updated)
        loaded = store.load("test-project")
        assert loaded.system_prompt == "new prompt"
        assert loaded.project_spec is None

    def test_delete_existing(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        store.save(_full_config())
        assert store.delete("test-project") is True
        assert store.load("test-project") is None

    def test_delete_nonexistent(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        assert store.delete("nope") is False

    def test_list_projects(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        store.save(DomainConfig(project_name="alpha"))
        store.save(DomainConfig(project_name="beta"))
        store.save(DomainConfig(project_name="gamma"))
        assert store.list_projects() == ["alpha", "beta", "gamma"]

    def test_exists(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        assert store.exists("test-project") is False
        store.save(_full_config())
        assert store.exists("test-project") is True

    def test_empty_name_raises(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        with pytest.raises(ValueError, match="must not be empty"):
            store.save(DomainConfig(project_name=""))

    def test_creates_directory(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "dir"
        store = FileProjectStore(target)
        store.save(DomainConfig(project_name="x"))
        assert target.exists()
        assert store.load("x") is not None

    def test_file_is_valid_json(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        store.save(_full_config())
        path = tmp_path / "projects" / "test-project.json"
        data = json.loads(path.read_text())
        assert data["project_name"] == "test-project"
        assert isinstance(data["project_spec"], dict)


# ═════════════════════════════════════════════════════════════════════
#  End-to-end: File → DomainConfig → ArchitecturalOntologyContext
# ═════════════════════════════════════════════════════════════════════


class TestFileStoreEndToEnd:
    def test_save_load_build_context(self, tmp_path):
        store = FileProjectStore(tmp_path / "projects")
        store.save(_full_config())

        loaded = store.load("test-project")
        assert loaded is not None

        ctx = ArchitecturalOntologyContext(loaded)
        assert ctx.graph.node_count() >= 1
        assert len(ctx.rules) == 1
        assert ctx.system_prompt == "You are a test assistant."
        assert len(ctx.preflight_quiz) == 1
        assert ctx.preflight_quiz[0].question == "What node types exist?"
