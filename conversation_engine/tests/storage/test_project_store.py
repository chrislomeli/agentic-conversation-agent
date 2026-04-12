"""
Tests for DomainConfig and ProjectStore (InMemoryProjectStore).

Coverage:
- DomainConfig: creation, frozen semantics, all-None fields, metadata
- InMemoryProjectStore: save, load, delete, list, exists, upsert, empty name
- ArchitecturalOntologyContext.from_config: round-trip from DomainConfig
"""

from __future__ import annotations

import pytest

from conversation_engine.graph.architectural_context import (
    ArchitecturalOntologyContext,
)
from conversation_engine.infrastructure.llm.architectural_quiz import (
    ARCHITECTURAL_QUIZ,
    ARCHITECTURAL_SYSTEM_PROMPT,
)
from conversation_engine.models import Goal, Requirement
from conversation_engine.models.base import BaseEdge
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.project_spec import ProjectSpecification
from conversation_engine.models.rule_node import IntegrityRule

# from conversation_engine.models.query_node import GraphQueryPattern
from conversation_engine.models.validation_quiz import FactualQuiz
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.project_store import (
    InMemoryProjectStore,
    ProjectStore,
)
from conversation_engine.storage.snapshot_facade import graph_to_snapshot

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
            id="rule-goal-req",
            name="Goal → Requirement",
            description="Every goal must have at least one requirement",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            edge_type="SATISFIED_BY",
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
            expected_answer="goal, requirement",
            weight=1.0,
            min_score=0.5,
        ),
    ]


def _sample_spec() -> ProjectSpecification:
    return graph_to_snapshot("test-project", _sample_graph())


def _full_config(**overrides) -> DomainConfig:
    defaults = dict(
        project_name="test-project",
        project_spec=_sample_spec(),
        rules=_sample_rules(),
        quiz=_sample_quiz(),
        # query_patterns=[],
        system_prompt="You are a test assistant.",
        metadata={"version": "1.0"},
    )
    defaults.update(overrides)
    return DomainConfig(**defaults)


# ═════════════════════════════════════════════════════════════════════
#  DomainConfig Tests
# ═════════════════════════════════════════════════════════════════════


class TestDomainConfig:
    def test_create_full_config(self):
        cfg = _full_config()
        assert cfg.project_name == "test-project"
        assert cfg.project_spec is not None
        assert len(cfg.project_spec.goals) == 1
        assert len(cfg.project_spec.requirements) == 1
        assert len(cfg.rules) == 1
        assert len(cfg.quiz) == 1
        assert cfg.system_prompt == "You are a test assistant."
        assert cfg.metadata == {"version": "1.0"}

    def test_create_minimal_config(self):
        cfg = DomainConfig(project_name="bare")
        assert cfg.project_name == "bare"
        assert cfg.project_spec is None
        assert cfg.rules is None
        assert cfg.quiz is None
        # assert cfg.query_patterns is None
        assert cfg.system_prompt is None
        assert cfg.metadata == {}

    def test_frozen(self):
        cfg = _full_config()
        with pytest.raises(AttributeError):
            # noinspection PyDataclass
            cfg.project_name = "changed"  # type: ignore[misc]

    def test_metadata_defaults_to_empty_dict(self):
        cfg = DomainConfig(project_name="x")
        assert cfg.metadata == {}


# ═════════════════════════════════════════════════════════════════════
#  InMemoryProjectStore Tests
# ═════════════════════════════════════════════════════════════════════


class TestInMemoryProjectStore:
    def test_is_project_store(self):
        store = InMemoryProjectStore()
        assert isinstance(store, ProjectStore)

    # ── save / load ──────────────────────────────────────────────────

    def test_save_and_load(self):
        store = InMemoryProjectStore()
        cfg = _full_config()
        store.save(cfg)
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.project_name == "test-project"
        assert loaded.project_spec is not None
        assert len(loaded.project_spec.goals) == 1

    def test_load_nonexistent_returns_none(self):
        store = InMemoryProjectStore()
        assert store.load("nope") is None

    def test_save_upsert_replaces(self):
        store = InMemoryProjectStore()
        store.save(_full_config())
        new_cfg = DomainConfig(
            project_name="test-project",
            system_prompt="updated prompt",
        )
        store.save(new_cfg)
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.system_prompt == "updated prompt"
        assert loaded.project_spec is None  # overwritten, not merged

    def test_save_empty_name_raises(self):
        store = InMemoryProjectStore()
        cfg = DomainConfig(project_name="")
        with pytest.raises(ValueError, match="must not be empty"):
            store.save(cfg)

    # ── delete ───────────────────────────────────────────────────────

    def test_delete_existing(self):
        store = InMemoryProjectStore()
        store.save(_full_config())
        assert store.delete("test-project") is True
        assert store.load("test-project") is None

    def test_delete_nonexistent(self):
        store = InMemoryProjectStore()
        assert store.delete("nope") is False

    # ── list ─────────────────────────────────────────────────────────

    def test_list_projects_empty(self):
        store = InMemoryProjectStore()
        assert store.list_projects() == []

    def test_list_projects_multiple(self):
        store = InMemoryProjectStore()
        store.save(DomainConfig(project_name="alpha"))
        store.save(DomainConfig(project_name="beta"))
        store.save(DomainConfig(project_name="gamma"))
        names = sorted(store.list_projects())
        assert names == ["alpha", "beta", "gamma"]

    # ── exists ───────────────────────────────────────────────────────

    def test_exists_true(self):
        store = InMemoryProjectStore()
        store.save(_full_config())
        assert store.exists("test-project") is True

    def test_exists_false(self):
        store = InMemoryProjectStore()
        assert store.exists("nope") is False

    # ── round-trip with all fields ───────────────────────────────────

    def test_round_trip_preserves_all_fields(self):
        store = InMemoryProjectStore()
        original = _full_config()
        store.save(original)
        loaded = store.load("test-project")
        assert loaded is not None
        assert loaded.project_name == original.project_name
        assert loaded.system_prompt == original.system_prompt
        assert loaded.metadata == original.metadata
        assert len(loaded.rules) == len(original.rules)
        assert len(loaded.quiz) == len(original.quiz)


# ═════════════════════════════════════════════════════════════════════
#  ArchitecturalOntologyContext(DomainConfig) Tests
# ═════════════════════════════════════════════════════════════════════


class TestArchitecturalContextFromConfig:
    def test_from_full_config(self):
        cfg = _full_config()
        ctx = ArchitecturalOntologyContext(cfg)
        # graph is built at runtime from the spec
        assert ctx.graph.node_count() >= 1
        assert len(ctx.rules) == 1
        assert ctx.system_prompt == "You are a test assistant."
        assert len(ctx.preflight_quiz) == 1
        assert ctx.preflight_quiz[0].question == "What node types exist?"

    def test_from_minimal_config_uses_defaults(self):
        cfg = DomainConfig(project_name="bare")
        ctx = ArchitecturalOntologyContext(cfg)
        assert ctx.graph.node_count() == 0
        assert ctx.rules == []
        assert ctx.system_prompt == ARCHITECTURAL_SYSTEM_PROMPT
        assert ctx.preflight_quiz == list(ARCHITECTURAL_QUIZ)

    def test_custom_system_prompt_overrides_default(self):
        cfg = DomainConfig(
            project_name="custom",
            system_prompt="Custom prompt here.",
        )
        ctx = ArchitecturalOntologyContext(cfg)
        assert ctx.system_prompt == "Custom prompt here."

    def test_custom_quiz_overrides_default(self):
        custom_quiz = [
            FactualQuiz(
                question="Custom question?",
                expected_answer="custom",
            ),
        ]
        cfg = DomainConfig(project_name="custom", quiz=custom_quiz)
        ctx = ArchitecturalOntologyContext(cfg)
        assert len(ctx.preflight_quiz) == 1
        assert ctx.preflight_quiz[0].question == "Custom question?"

    def test_none_quiz_falls_back_to_architectural(self):
        cfg = DomainConfig(project_name="no-quiz", quiz=None)
        ctx = ArchitecturalOntologyContext(cfg)
        assert len(ctx.preflight_quiz) == len(ARCHITECTURAL_QUIZ)

    def test_empty_quiz_list_means_no_preflight(self):
        cfg = DomainConfig(project_name="skip-preflight", quiz=[])
        ctx = ArchitecturalOntologyContext(cfg)
        assert ctx.preflight_quiz == []


# ═════════════════════════════════════════════════════════════════════
#  End-to-end: Store → Load → Context
# ═════════════════════════════════════════════════════════════════════


class TestStoreToContext:
    def test_save_load_and_build_context(self):
        store = InMemoryProjectStore()
        store.save(_full_config())

        loaded = store.load("test-project")
        assert loaded is not None

        ctx = ArchitecturalOntologyContext(loaded)
        assert ctx.graph.node_count() >= 1
        assert len(ctx.rules) == 1
        assert ctx.system_prompt == "You are a test assistant."

    def test_load_missing_project_returns_none(self):
        store = InMemoryProjectStore()
        assert store.load("does-not-exist") is None
