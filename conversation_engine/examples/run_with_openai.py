#!/usr/bin/env python
"""
Driver: run the full conversation graph with a real OpenAI LLM.

Steps through:
  1. Build the ArchitecturalOntologyContext with a graph that has gaps
  2. Create an OpenAI-backed CallLLM adapter
  3. Build the conversation graph (with logging + metrics middleware)
  4. Invoke the graph — preflight validates the LLM, then the loop runs
  5. Print results at each stage

Usage:
    # Set your API key
    export OPENAI_API_KEY="sk-..."

    # Run the driver (from repo root)
    python examples/run_with_openai.py

    # Or use a specific model
    OPENAI_MODEL=gpt-4o python examples/run_with_openai.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Setup path (so this script works from repo root) ─────────────────
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

from conversation_engine.graph.architectural_context import ArchitecturalOntologyContext
from conversation_engine.graph.builder import build_conversation_graph
from conversation_engine.infrastructure.llm import make_openai_llm
from conversation_engine.infrastructure.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
)
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.storage.snapshot_facade import graph_to_snapshot
from ogar.fixtures import create_graph_with_gaps

# ── 1. Build domain context ─────────────────────────────────────────


def build_context() -> ArchitecturalOntologyContext:
    """Create an architectural context with a graph that has integrity gaps."""
    graph = create_graph_with_gaps()

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

    spec = graph_to_snapshot("architectural-demo", graph)
    config = DomainConfig(project_name="architectural-demo", project_spec=spec, rules=rules)
    ctx = ArchitecturalOntologyContext(config)
    print(f"  Graph: {ctx.graph.node_count()} nodes, {ctx.graph.edge_count()} edges")
    print(f"  Rules: {len(rules)}")
    print(f"  Preflight quiz: {len(ctx.preflight_quiz)} questions")
    return ctx


# ── 2. Create LLM ───────────────────────────────────────────────────


def create_llm():
    """Create an OpenAI-backed LLM callable."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"  Model: {model}")
    llm = make_openai_llm(model=model)
    print(f"  Adapter: {type(llm).__name__}")
    return llm


# ── 3. Build and run graph ──────────────────────────────────────────


def run_graph(ctx, llm):
    """Build the graph with middleware and invoke it."""
    metrics = MetricsMiddleware()
    graph = build_conversation_graph(
        node_middleware=[LoggingMiddleware(), metrics],
    )

    initial_state = {
        "context": ctx,
        "session_id": "openai-driver-session",
        "llm": llm,
        "findings": [],
        "messages": [],
        "current_turn": 0,
        "status": "running",
        "node_result": None,
        "preflight_passed": False,
    }

    print("\n  Invoking graph...")
    result = graph.invoke(initial_state)
    return result, metrics


# ── 4. Print results ────────────────────────────────────────────────


def print_results(result, metrics):
    """Display the outcome."""

    print(f"\n  Status: {result['status']}")
    print(f"  Preflight passed: {result.get('preflight_passed')}")
    print(f"  Turns completed: {result['current_turn']}")

    # Findings
    findings = result.get("findings", [])
    open_findings = [f for f in findings if not f.resolved]
    print(f"\n  Total findings: {len(findings)}")
    print(f"  Open findings: {len(open_findings)}")
    for f in open_findings[:5]:
        print(f"    [{f.severity}] {f.message}")

    # Messages from the LLM
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if m.type == "ai"]
    print(f"\n  AI messages: {len(ai_messages)}")
    for i, msg in enumerate(ai_messages):
        preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"\n  --- Message {i + 1} ---")
        print(f"  {preview}")

    # Metrics
    snap = metrics.snapshot()
    print("\n  Node execution counts:")
    for node_name, data in sorted(snap.items()):
        avg_ms = data["avg_duration"] * 1000
        print(
            f"    {node_name}: {data['call_count']}x, "
            f"avg {avg_ms:.0f}ms, "
            f"errors: {data['error_count']}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Conversation Graph Driver — Real OpenAI LLM")
    print("=" * 60)

    print("\n[1] Building domain context...")
    ctx = build_context()

    print("\n[2] Creating OpenAI LLM...")
    try:
        llm = create_llm()
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        print("  Set OPENAI_API_KEY and try again.")
        sys.exit(1)

    print("\n[3] Running conversation graph...")
    print("  Topology: START → preflight → validate → converse → route")
    result, metrics = run_graph(ctx, llm)

    print("\n[4] Results:")
    print_results(result, metrics)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
