"""
Shared fixtures for tests and examples.

Graph fixtures: pre-built KnowledgeGraph instances with specific shapes.
Config fixtures: reusable rules, DomainConfigs, contexts, and state factories.
"""

from conversation_engine.fixtures.config_fixtures import (
    goal_req_rule,
    make_context,
    make_state,
    minimal_state,
    partial_config,
    req_step_rule,
    sample_config,
    standard_rules,
)
from conversation_engine.fixtures.graph_fixtures import (
    create_graph_complete,
    create_graph_partial_coverage,
    create_graph_with_gaps,
    create_graph_with_orphans,
    create_minimal_graph,
)
from conversation_engine.fixtures.project_fixtures import (
    conversation_engine_meta_spec,
    infer_meta_spec_from_paths,
    load_project_spec_fixture_json,
)

__all__ = [
    # Graph fixtures
    "create_graph_with_gaps",
    "create_graph_with_orphans",
    "create_graph_complete",
    "create_graph_partial_coverage",
    "create_minimal_graph",
    # Config fixtures
    "goal_req_rule",
    "req_step_rule",
    "standard_rules",
    "sample_config",
    "partial_config",
    "make_context",
    "minimal_state",
    "make_state",
    # ProjectSpecification fixtures
    "conversation_engine_meta_spec",
    "infer_meta_spec_from_paths",
    "load_project_spec_fixture_json",
]
