"""
Tests for the resolve_domain graph node and route_after_resolve_domain router.

Coverage:
- Scenario 1: context already provided → pass through
- Scenario 2a: project_name + store → load and build context
- Scenario 2b: project_name + store, project not found → error
- Scenario 3: no project_name, human provides one → needs_project_name
- Fallback: no context, no name, no human → error
- Router: running → preflight, needs_project_name → resolve_domain, error → __end__
"""

from __future__ import annotations

import pytest

from conversation_engine.graph.nodes import resolve_domain
from conversation_engine.graph.builder import route_after_resolve_domain
from conversation_engine.graph.architectural_context import (
    ArchitecturalOntologyContext,
)
from conversation_engine.storage.project_store import InMemoryProjectStore
from conversation_engine.infrastructure.human import MockHuman
from conversation_engine.fixtures import (
    sample_config,
    partial_config,
    minimal_state,
)


# ═════════════════════════════════════════════════════════════════════
#  resolve_domain node
# ═════════════════════════════════════════════════════════════════════


class TestResolveDomainPassThrough:
    """Scenario 1: context already provided."""

    def test_context_set_returns_running(self):
        config = sample_config()
        ctx = ArchitecturalOntologyContext(config)
        state = minimal_state(context=ctx)
        result = resolve_domain(state)
        assert result["status"] == "running"
        assert "context" not in result  # no change needed

    def test_context_set_does_not_overwrite(self):
        config = sample_config()
        ctx = ArchitecturalOntologyContext(config)
        state = minimal_state(
            context=ctx, project_name="other", project_store=InMemoryProjectStore()
        )
        result = resolve_domain(state)
        assert result["status"] == "running"
        assert "context" not in result


class TestResolveDomainFromStore:
    """Scenario 2: load from store by project_name."""

    def test_load_partial_project(self):
        store = InMemoryProjectStore()
        config = partial_config()
        store.save(config)
        state = minimal_state(project_name="test-project", project_store=store)
        result = resolve_domain(state)

        assert result["status"] == "running"
        assert "context" in result
        ctx = result["context"]
        assert isinstance(ctx, ArchitecturalOntologyContext)
        assert ctx.graph.node_count() == 2
        assert len(ctx.rules) == 1
        assert ctx.system_prompt == "Test system prompt."

    def test_load_existing_project(self):
        store = InMemoryProjectStore()
        config = sample_config()
        store.save(config)
        state = minimal_state(project_name="test-project", project_store=store)
        result = resolve_domain(state)

        assert result["status"] == "running"
        assert "context" in result
        ctx = result["context"]
        assert isinstance(ctx, ArchitecturalOntologyContext)
        assert ctx.graph.node_count() == 2
        assert len(ctx.rules) == 1
        assert ctx.system_prompt == "Test system prompt."

    def test_project_not_found_returns_error(self):
        store = InMemoryProjectStore()
        state = minimal_state(project_name="nonexistent", project_store=store)
        result = resolve_domain(state)

        assert result["status"] == "error"
        assert len(result["messages"]) == 1
        assert "not found" in result["messages"][0].content


class TestResolveDomainAskHuman:
    """Scenario 3: ask human for project name."""

    def test_human_provides_name(self):
        human = MockHuman(responses=["my-project"])
        state = minimal_state(human=human)
        result = resolve_domain(state)

        assert result["status"] == "needs_project_name"
        assert result["project_name"] == "my-project"

    def test_human_provides_name_with_whitespace(self):
        human = MockHuman(responses=["  padded-name  "])
        state = minimal_state(human=human)
        result = resolve_domain(state)

        assert result["project_name"] == "padded-name"


class TestResolveDomainFallback:
    """Fallback: nothing to work with."""

    def test_no_context_no_name_no_human(self):
        state = minimal_state()
        result = resolve_domain(state)
        assert result["status"] == "error"
        assert "Cannot start" in result["messages"][0].content

    def test_no_context_no_name_no_store(self):
        state = minimal_state(project_name="test-project")
        result = resolve_domain(state)
        assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════
#  route_after_resolve_domain
# ═════════════════════════════════════════════════════════════════════


class TestRouteAfterResolveDomain:
    def test_running_goes_to_preflight(self):
        assert route_after_resolve_domain({"status": "running"}) == "preflight"

    def test_needs_project_name_loops_back(self):
        assert route_after_resolve_domain({"status": "needs_project_name"}) == "resolve_domain"

    def test_error_goes_to_end(self):
        assert route_after_resolve_domain({"status": "error"}) == "__end__"

    def test_default_status_goes_to_preflight(self):
        assert route_after_resolve_domain({}) == "preflight"


# ═════════════════════════════════════════════════════════════════════
#  Integration: resolve_domain → router → preflight chain
# ═════════════════════════════════════════════════════════════════════


class TestResolveDomainIntegration:
    """Simulate the resolve → route → resolve loop for scenario 3."""

    def test_human_name_then_store_load(self):
        store = InMemoryProjectStore()
        store.save(sample_config())
        human = MockHuman(responses=["test-project"])

        # Turn 1: no project_name, human provides one
        state = minimal_state(project_store=store, human=human)
        result1 = resolve_domain(state)
        assert result1["status"] == "needs_project_name"
        assert result1["project_name"] == "test-project"

        route1 = route_after_resolve_domain(result1)
        assert route1 == "resolve_domain"

        # Turn 2: project_name now set, load from store
        state.update(result1)
        result2 = resolve_domain(state)
        assert result2["status"] == "running"
        assert isinstance(result2["context"], ArchitecturalOntologyContext)

        route2 = route_after_resolve_domain(result2)
        assert route2 == "preflight"
