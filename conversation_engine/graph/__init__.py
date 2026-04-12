"""
Conversation graph — LangGraph orchestration layer.

This package wires a domain-agnostic conversation loop that delegates
all domain logic to an injected ConversationContext.

The loop topology (preflight → validate → converse → route) is fixed.
The *domain* is pluggable: implement ConversationContext for your
domain and pass it in via state.

Infrastructure is injected at build time:
- interceptors / middleware via build_conversation_graph(...)
- LLM callable via state["llm"]
- Human surface via state["human"]
- Tool client via state["tool_client"]
- Error handling via ErrorHandlingMiddleware
"""

from conversation_engine.graph.context import (
    ConversationContext,
    Finding,
    ValidationResult,
)
from conversation_engine.graph.state import (
    ConversationInput,
    ConversationOutput,
    ConversationState,
)
from conversation_engine.graph.builder import build_conversation_graph
from conversation_engine.graph.architectural_context import (
    ArchitecturalOntologyContext,
)
from conversation_engine.models.domain_config import DomainConfig

__all__ = [
    "ConversationContext",
    "Finding",
    "ValidationResult",
    "ConversationInput",
    "ConversationOutput",
    "ConversationState",
    "build_conversation_graph",
    "ArchitecturalOntologyContext",
    "DomainConfig",
]
