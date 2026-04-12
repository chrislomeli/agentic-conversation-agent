"""
LLM interaction protocols — the contract between graph nodes and LLM backends.

The conversation loop never imports openai, langchain_openai, or any
specific LLM library.  It calls through these protocols.  Swap the
implementation at graph construction time.

Design principles:
- Protocol, not ABC — structural subtyping, matches the ConversationContext pattern
- Request/Response are frozen dataclasses — immutable, serializable
- The protocol is synchronous — async support can wrap it later
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    """
    A structured request to an LLM.

    Domain-agnostic: the caller provides system prompt, user message,
    and optional structured context.  The LLM implementation decides
    how to format these for its specific backend.
    """

    system_prompt: str
    user_message: str
    context: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.2
    max_tokens: int | None = None
    response_format: str | None = None  # "json", "text", or None


@dataclass(frozen=True)
class LLMResponse:
    """
    A structured response from an LLM.

    Always contains the raw text.  Optionally contains parsed
    structured data if the LLM was asked for JSON output.
    """

    content: str
    structured: dict[str, Any] | None = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, etc.
    success: bool = True
    error: str | None = None


@runtime_checkable
class CallLLM(Protocol):
    """
    Interface: send a request to an LLM and get a response.

    Any callable matching this signature satisfies the protocol.
    This enables:
      - Deterministic stubs for testing
      - Real OpenAI / Anthropic / local model backends
      - ReAct agents that loop internally with tools
      - Pre-run validators that quiz the LLM before trusting it
    """

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Send a request to an LLM and return the response."""
        ...
