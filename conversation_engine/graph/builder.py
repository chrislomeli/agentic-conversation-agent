"""
Conversation subgraph builder.

Topology:
    START → resolve_domain → route_after_resolve_domain →
        preflight → route_after_preflight → validate → converse → route
    route_after_resolve_domain → preflight        (context ready)
    route_after_resolve_domain → resolve_domain   (human gave name, loop back)
    route_after_resolve_domain → END              (error)
    route_after_preflight      → validate         (preflight passes or was skipped)
    route_after_preflight      → END              (preflight fails → status='error')
    route_after_converse       → validate         (open findings remain, turns < max)
    route_after_converse       → END              (complete, max turns, error, or hand_off)

The `converse` node is a collaborative AI/human exchange:
  1. LLM analyzes findings + message history → produces AI message
  2. Human surface (CallHuman) collects the user's reply
  3. Both messages are appended to state, turn counter advances

The graph is **domain-agnostic**.  It works with Findings (not
Assessments) and delegates all domain logic to the injected
ConversationContext.

Infrastructure:
- InstrumentedGraph wraps every node with a composable NodeMiddleware chain
- Cross-cutting concerns (logging, metrics, error handling, retry, circuit
  breaker, config) are injected via node_middleware, not baked into topology
- Pre-flight LLM validation is a first-class node, not imperative build code
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence

from langgraph.graph import START, END

from conversation_engine.graph.state import ConversationState
from conversation_engine.graph.nodes import resolve_domain, preflight, validate, converse
from conversation_engine.infrastructure.instrumented_graph import (
    InstrumentedGraph,
    Interceptor,
    Middleware,
)
from conversation_engine.infrastructure.middleware.base import NodeMiddleware

logger = logging.getLogger(__name__)


MAX_TURNS = 5


# ── Routers ──────────────────────────────────────────────────────────


def route_after_resolve_domain(
    state: ConversationState,
) -> Literal["preflight", "resolve_domain", "__end__"]:
    """
    After resolve_domain:
    - "running"             → context is ready, proceed to preflight
    - "needs_project_name"  → human gave a name, loop back to resolve
    - "error"               → exit
    """
    status = state.get("status", "running")
    if status == "error":
        return "__end__"
    if status == "needs_project_name":
        return "resolve_domain"
    return "preflight"


def route_after_preflight(state: ConversationState) -> Literal["validate", "__end__"]:
    """
    After preflight: continue to validate if OK, exit if error.
    """
    if state.get("status") == "error":
        return "__end__"
    return "validate"


def route_after_converse(state: ConversationState) -> Literal["validate", "__end__"]:
    """
    Decide whether to loop back for another validation pass or finish.

    Exits when:
    - No open findings remain     →  "complete"
    - Max turns reached           →  "max_turns"
    - Status set to error/hand_off by a node  →  exit
    """
    status = state.get("status", "running")
    if status in ("complete", "error", "hand_off"):
        return "__end__"

    current_turn = state.get("current_turn", 0)
    if current_turn >= MAX_TURNS:
        return "__end__"

    open_findings = [f for f in state.get("findings", []) if not f.resolved]
    if not open_findings:
        return "__end__"

    return "validate"


# ── Builder ──────────────────────────────────────────────────────────


def build_conversation_graph(
    *,
    node_middleware: Sequence[NodeMiddleware] | None = None,
    # Legacy parameters — deprecated, kept for backwards compatibility
    interceptors: Sequence[Interceptor] | None = None,
    middleware: Sequence[Middleware] | None = None,
):
    """
    Build and compile the conversation subgraph.

    Parameters
    ----------
    node_middleware : Sequence[NodeMiddleware], optional
        Composable cross-cutting concerns applied to every node.
        Order matters: first in list = outermost wrapper.
        Typical order: Logging → Metrics → ErrorHandling → Retry → [node]
    interceptors : Sequence[Interceptor], optional
        DEPRECATED. Use node_middleware instead.
    middleware : Sequence[Middleware], optional
        DEPRECATED. Use node_middleware instead.

    Returns a compiled graph ready for .invoke() or .stream().

    Topology: START → preflight → validate → converse → route → ...
    The converse node handles both LLM reasoning and human interaction.
    """
    builder = InstrumentedGraph(
        ConversationState,
        node_middleware=node_middleware,
        interceptors=interceptors,
        middleware=middleware,
    )

    # Nodes
    builder.add_node("resolve_domain", resolve_domain)
    builder.add_node("preflight", preflight)
    builder.add_node("validate", validate)
    builder.add_node("converse", converse)

    # Edges
    builder.add_edge(START, "resolve_domain")
    builder.add_conditional_edges("resolve_domain", route_after_resolve_domain)
    builder.add_conditional_edges("preflight", route_after_preflight)
    builder.add_edge("validate", "converse")
    builder.add_conditional_edges("converse", route_after_converse)

    compiled_graph = builder.compile()
    return compiled_graph
