"""
Conversation graph state definitions.

Three separate schemas keep concerns clean:
- ConversationInput  — what the parent graph passes in (the contract boundary)
- ConversationOutput — what the subgraph returns (the contract boundary)
- ConversationState  — the internal working state (never crosses the boundary)

Design principles:
- Domain-agnostic — the state knows about Findings, not Assessments
- The ConversationContext protocol is the only bridge to domain specifics
- Minimal fields — only add what a node actually reads or writes
- The subgraph does NOT own the checkpointer, LLM client, or human surface
- Grows organically as new nodes demand new fields
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from conversation_engine.graph.context import ConversationContext, Finding
from conversation_engine.infrastructure.node_validation import NodeResult
from conversation_engine.infrastructure.llm import CallLLM
from conversation_engine.infrastructure.human import CallHuman
from conversation_engine.infrastructure.tool_client import ToolClient
from conversation_engine.storage.project_store import ProjectStore


# ── Subgraph contract ────────────────────────────────────────────────


class ConversationInput(TypedDict):
    """What the parent graph passes in at entry."""

    context: ConversationContext
    session_id: str


class ConversationOutput(TypedDict):
    """What the subgraph returns at exit."""

    findings: List[Finding]
    domain_state: Dict[str, Any]
    session_summary: str
    exit_reason: Literal["complete", "hand_off", "error", "max_turns"]


# ── Internal working state ───────────────────────────────────────────


class ConversationState(TypedDict):
    # Injected domain context — set by resolve_domain, read by all downstream nodes
    context: Optional[ConversationContext]
    session_id: str

    # Domain resolution inputs — resolve_domain uses these to build context
    project_name: Optional[str]
    project_store: Optional[ProjectStore]

    # Injected LLM callable — optional, nodes fall back to stub if absent
    llm: Optional[CallLLM]

    # Injected human surface — optional, nodes skip human interaction if absent
    human: Optional[CallHuman]

    # Injected tool client — optional, converse node uses it for ReAct agent loop
    tool_client: Optional[ToolClient]

    # Built during conversation (domain-agnostic)
    findings: List[Finding]
    messages: Annotated[List[BaseMessage], add_messages]
    current_turn: int

    # Control flow
    status: Literal["running", "interrupted", "complete", "hand_off", "error"]

    # Node validation — populated by validated_node decorator or nodes directly
    node_result: Optional[NodeResult]

    # Pre-flight LLM validation — set to True after first successful pass
    preflight_passed: bool
