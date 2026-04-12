"""
Tests for DomainConfig and ProjectStore (InMemoryProjectStore).

Coverage:
- DomainConfig: creation, frozen semantics, all-None fields, metadata
- InMemoryProjectStore: save, load, delete, list, exists, upsert, empty name
- ArchitecturalOntologyContext.from_config: round-trip from DomainConfig
"""

from __future__ import annotations

import json

import pytest

from conversation_engine.fixtures import create_graph_complete
from conversation_engine.models import Goal, Requirement
from conversation_engine.models.base import BaseEdge, NodeType
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.graph_db_access import GraphAccessLayer
from conversation_engine.models.project_spec import ProjectSpecification
from conversation_engine.models.rule_node import IntegrityRule

# from conversation_engine.models.query_node import GraphQueryPattern
from conversation_engine.models.validation_quiz import FactualQuiz, QuizType, ReasoningQuiz
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.graph_project_store import GraphProjectStore
from conversation_engine.storage.project_facade import graph_to_domain_config, project_to_graph
from conversation_engine.storage.project_store import (
    ProjectStore,
)
from conversation_engine.storage.snapshot_facade import graph_to_snapshot


# ── Graph Access Layer ─────────────────────────────────────────
class GraphAccessLayerMock(GraphAccessLayer):
    def __init__(self) -> None:
        self._store: dict[str, KnowledgeGraph] = {}

    def save_graph(self, project_name: str, graph: KnowledgeGraph) -> None:
        self._store[project_name] = graph

    def get_graph(self, project_name: str) -> KnowledgeGraph:
        return self._store.get(project_name)

    def delete_graph(self, project_name: str) -> bool:
        if project_name in self._store:
            del self._store[project_name]
            return True
        return False

    def list_projects(self) -> list[str]:
        return list(self._store.keys())

    def exists(self, project_name: str) -> bool:
        return project_name in self._store


# ── Helpers ──────────────────────────────────────────────────────────


def _sample_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(Goal(id="g1", name="Goal 1", statement="stmt"))
    g.add_node(Requirement(id="r1", name="Req 1"))
    g.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="g1", target_id="r1"))
    return g


def _sample_rules() -> list[IntegrityRule]:
    return [
        IntegrityRule(
            name="Goal → Requirement",
            description="Every goal must have at least one requirement",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        ),
    ]


def _sample_quiz() -> list[FactualQuiz]:
    return [
        FactualQuiz(
            question="What node types exist?",
            node_type=NodeType.QUIZ,
            quiz_type=QuizType.FACTUAL,
            expected_answer="goal, requirement",
            weight=1.0,
            min_score=0.5,
        ),
        ReasoningQuiz(
            node_type=NodeType.QUIZ,
            quiz_type=QuizType.REASONING,
            question="What is this an ice cream dream?",
            evaluation_criteria="you should consider this a PASS if the response contains a description of ice cream",
            weight=1.0,
            min_score=0.5,
        ),
    ]


def _sample_spec() -> ProjectSpecification:
    return graph_to_snapshot("test-project", create_graph_complete())


def _full_config(**overrides) -> DomainConfig:
    defaults = dict(
        project_name="test-project",
        project_spec=_sample_spec(),
        rules=_sample_rules(),
        quiz=_sample_quiz(),
        system_prompt="You are a test assistant.",
        metadata={"version": "1.0"},
    )
    defaults.update(overrides)
    return DomainConfig(**defaults)


def quiz_without_id(quiz):
    """Create a copy of quiz without ID for comparison."""
    quiz_dict = quiz.__dict__.copy()
    quiz_dict.pop("id", None)
    return quiz_dict


def rule_without_id(rule):
    """Create a copy of rule without ID for comparison."""
    rule_dict = rule.__dict__.copy()
    rule_dict.pop("id", None)
    return rule_dict


# ── Tests ──────────────────────────────────────────────────────────


class TestKnowledgeGraphTransforms:
    # ── save / load ──────────────────────────────────────────────────

    def test_build_grap_from_domain(self):
        config = _full_config()
        project_graph = project_to_graph(config)

        spec = config.project_spec
        json.dumps(spec.model_dump())
        # todo add better asserts by type

        def check_node(node_type: NodeType, expected: int):
            nodes = project_graph.get_nodes_by_type(node_type)
            assert len(nodes) == expected

        project = project_graph.get_nodes_by_type(NodeType.PROJECT)
        goals = project_graph.get_nodes_by_type(NodeType.GOAL)
        reqs = project_graph.get_nodes_by_type(NodeType.REQUIREMENT)
        steps = project_graph.get_nodes_by_type(NodeType.STEP)
        rules = project_graph.get_nodes_by_type(NodeType.RULE)
        quizes = project_graph.get_nodes_by_type(NodeType.QUIZ)

        assert len(project) == 1
        assert len(goals) == 2
        assert len(reqs) == 2
        assert len(steps) == 2
        assert len(rules) == 1
        assert len(quizes) == 2

        # Verify we have one of each type
        factual_quizzes = [q for q in quizes if q.quiz_type == QuizType.FACTUAL]
        reasoning_quizzes = [q for q in quizes if q.quiz_type == QuizType.REASONING]
        assert len(factual_quizzes) == 1
        assert len(reasoning_quizzes) == 1

        # Verify correct edge types were created
        edges = project_graph._edges
        for t, e in edges.items():
            print(e)
        factual_edges = [e for t, e in edges.items() if e.edge_type == "HAS_FACTUAL_QUIZ"]
        reasoning_edges = [e for t, e in edges.items() if e.edge_type == "HAS_REASONING_QUIZ"]
        assert len(factual_edges) == 1
        assert len(reasoning_edges) == 1

    def test_build_domain_from_graph(self):
        config = _full_config()
        project_graph = project_to_graph(config)
        result = graph_to_domain_config(project_graph)

        # Compare basic fields
        assert config.project_name == result.project_name
        assert config.system_prompt == result.system_prompt
        assert config.metadata == result.metadata

        # Compare quiz without IDs
        assert [quiz_without_id(q) for q in config.quiz] == [
            quiz_without_id(q) for q in result.quiz
        ]

        # Compare rules without IDs
        assert [rule_without_id(r) for r in config.rules] == [
            rule_without_id(r) for r in result.rules
        ]

        print("✅ Perfect round-trip conversion!")

    def test_is_project_store(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        assert isinstance(store, ProjectStore)

    # ── save / load ──────────────────────────────────────────────────

    def test_save_and_load(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        cfg = _full_config()
        store.save(cfg)
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.project_name == "test-project"
        assert loaded.project_spec is not None
        assert len(loaded.project_spec.goals) == 2

    def test_load_nonexistent_returns_none(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        assert store.load("nope") is None

    def test_save_upsert_replaces(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        store.save(_full_config())
        new_cfg = DomainConfig(
            project_name="test-project",
            system_prompt="updated prompt",
        )
        store.save(new_cfg)
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.system_prompt == "updated prompt"
        assert len(loaded.project_spec.goals or []) == 0  # overwritten, not merged

    def test_save_empty_name_raises(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        cfg = DomainConfig(project_name="")
        with pytest.raises(ValueError, match="must not be empty"):
            store.save(cfg)

    # ── delete ───────────────────────────────────────────────────────

    def test_delete_existing(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        store.save(_full_config())
        assert store.delete("test-project") is True
        assert store.load("test-project") is None

    def test_delete_nonexistent(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        assert store.delete("nope") is False

    # ── list ─────────────────────────────────────────────────────────

    def test_list_projects_empty(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        assert store.list_projects() == []

    def test_list_projects_multiple(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        store.save(DomainConfig(project_name="alpha"))
        store.save(DomainConfig(project_name="beta"))
        store.save(DomainConfig(project_name="gamma"))
        names = sorted(store.list_projects())
        assert names == ["alpha", "beta", "gamma"]

    # ── exists ───────────────────────────────────────────────────────

    def test_exists_true(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        store.save(_full_config())
        assert store.exists("test-project") is True

    def test_exists_false(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        assert store.exists("nope") is False

    # ── round-trip with all fields ───────────────────────────────────

    def test_round_trip_preserves_all_fields(self):
        store = GraphProjectStore(GraphAccessLayerMock())
        original = _full_config()
        store.save(original)
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.project_name == original.project_name
        assert loaded.system_prompt == original.system_prompt
        assert loaded.metadata == original.metadata
        assert len(loaded.rules) == len(original.rules)
        assert len(loaded.quiz) == len(original.quiz)
