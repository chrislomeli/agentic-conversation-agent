#!/usr/bin/env python
"""
Interactive chat driver: ReAct agent with tool-calling.

The LLM now has agency over the conversation via tools:
  - ask_human   — surface a message to the human, collect their reply
  - revalidate  — re-run integrity checks on the knowledge graph
  - mark_complete — signal that the conversation goal has been met

The graph topology is still:
  preflight → validate → converse(agent loop) → route

But inside the converse node, the LLM decides which tools to call and when.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/chat_with_openai.py

    # Or use a specific model
    OPENAI_MODEL=gpt-4o python examples/chat_with_openai.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from conversation_engine.graph import ConversationState
from conversation_engine.storage.snapshot_facade import graph_to_snapshot

# ── Setup path ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Load API keys from .env if not already set ───────────────────────
_env_file = Path.home() / "Source" / "SECRETS" / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value

from conversation_engine.fixtures import create_graph_complete
from conversation_engine.graph.architectural_context import ArchitecturalOntologyContext
from conversation_engine.graph.builder import build_conversation_graph
from conversation_engine.infrastructure.human import ConsoleHuman
from conversation_engine.infrastructure.llm import make_openai_llm
from conversation_engine.infrastructure.middleware import (
    MetricsMiddleware,
)
from conversation_engine.infrastructure.tool_client import (
    LocalToolClient,
    ToolRegistry,
    make_ask_human_tool,
    make_mark_complete_tool,
    make_revalidate_tool,
)
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.rule_node import IntegrityRule

# def _sample_config() -> DomainConfig:
#     g = KnowledgeGraph()
#     g.add_node(Goal(id="g1", name="Goal 1", statement="A goal"))
#     g.add_node(Requirement(id="r1", name="Req 1"))
#     g.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="g1", target_id="r1"))
#     return DomainConfig(
#         project_name="test-project",
#         project_spec=graph_to_snapshot("test-project", g),
#         quiz=list(ARCHITECTURAL_QUIZ),
#         rules=[
#             IntegrityRule(
#                 id="rule-goal-req",
#                 name="Goal → Requirement",
#                 description="Every goal must have at least one requirement",
#                 applies_to_node_type="goal",
#                 rule_type="minimum_outgoing_edge_count",
#                 target_node_types=["requirement"],
#                 minimum_count=1,
#                 severity="high",
#                 failure_message_template="Goal '{subject_name}' has no requirements.",
#             ),
#             IntegrityRule(
#                 id="rule-req-comp",
#                 name="Requirement → Component",
#                 description="Every requirement must have at least one component",
#                 applies_to_node_type="requirement",
#                 rule_type="minimum_outgoing_edge_count",
#                 target_node_types=["component"],
#                 minimum_count=1,
#                 severity="medium",
#                 failure_message_template="Requirement '{subject_name}' has no components.",
#             ),
#         ],
#         system_prompt="Test system prompt.",
#     )


def _minimal_state(**overrides) -> ConversationState:
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


def build_context() -> ArchitecturalOntologyContext:
    """Create an architectural context with a graph that has integrity gaps."""
    graph = create_graph_complete()
    rules = [
        IntegrityRule(
            id="rule-goal-req",
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
            id="rule-req-step",
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
    spec = graph_to_snapshot("architectural-chat", graph)
    config = DomainConfig(project_name="architectural-chat", project_spec=spec, rules=rules)
    return ArchitecturalOntologyContext(config)


def build_tool_client(human, ctx, findings_ref):
    """
    Build a LocalToolClient with conversation tools registered.

    Parameters
    ----------
    human : CallHuman
        The human surface implementation.
    ctx : ConversationContext
        The domain context for revalidation.
    findings_ref : list
        Mutable list used as a closure reference so the revalidate tool
        can access the latest findings.
    """
    registry = ToolRegistry()
    registry.register(make_ask_human_tool(human))
    registry.register(make_revalidate_tool(ctx, lambda: findings_ref[0]))
    registry.register(make_mark_complete_tool())
    return LocalToolClient(registry)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("  Architectural Integrity Chat (ReAct Agent)")
    print("  The AI will use tools to communicate with you")
    print("  Type 'quit' or 'exit' to leave")
    print("=" * 60)

    # ── Build context ──────────────────────────────────────────
    ctx = build_context()
    print(f"\n  Graph: {ctx.graph.node_count()} nodes, {ctx.graph.edge_count()} edges")

    # open wrapper class with llm
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        llm = make_openai_llm(model=model)
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    print(f"  LLM: {model}")

    human = ConsoleHuman(prompt_prefix="\nYou> ")

    # findings_ref is a mutable container so the revalidate tool closure
    # can access the latest findings from state
    findings_ref = [[]]
    tool_client = build_tool_client(human, ctx, findings_ref)

    print(f"  Tools: {', '.join(t['name'] for t in tool_client.list_tools())}")

    metrics = MetricsMiddleware()
    graph = build_conversation_graph(node_middleware=[metrics])

    # ── Initial state ─────────────────────────────────────────────
    state = {
        "context": ctx,
        "session_id": "chat-session",
        "llm": llm,
        "human": None,  # human interaction now goes through ask_human tool
        "tool_client": tool_client,
        "findings": [],
        "messages": [],
        "current_turn": 0,
        "status": "running",
        "node_result": None,
        "preflight_passed": False,
    }

    print("\n  Running preflight validation...")
    print("  Topology: preflight → validate → converse(agent) → route")
    print("  (The AI decides when to talk to you via the ask_human tool)\n")
    print("-" * 60)

    # ── Run the graph ─────────────────────────────────────────────
    result = graph.invoke(state)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Status: {result['status']}")
    print(f"  Preflight passed: {result.get('preflight_passed')}")
    print(f"  Turns completed: {result['current_turn']}")

    findings = result.get("findings", [])
    open_findings = [f for f in findings if not f.resolved]
    print(f"  Open findings: {len(open_findings)}")

    snap = metrics.snapshot()
    print("\n  Node metrics:")
    for node_name, data in sorted(snap.items()):
        avg_ms = data["avg_duration"] * 1000
        print(f"    {node_name}: {data['call_count']}x, avg {avg_ms:.0f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
