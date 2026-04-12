"""
Human interaction protocols — the contract between graph nodes and human surfaces.

Symmetric with the LLM protocol layer.  The conversation loop never
imports readline, flask, or any specific UI library.  It calls through
these protocols.  Swap the implementation at graph construction time.

Design principles:
- Protocol, not ABC — structural subtyping, matches CallLLM pattern
- Request/Response are frozen dataclasses — immutable, serializable
- The protocol is synchronous — async support can wrap it later
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class HumanRequest:
    """
    A structured request for human input.

    The node tells the human what it needs; the surface implementation
    decides how to present it (CLI prompt, web form, Slack message, etc.).
    """

    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    options: list[str] | None = None  # optional multiple-choice
    allow_skip: bool = True  # human can skip / say nothing


@dataclass(frozen=True)
class HumanResponse:
    """
    A structured response from a human.

    Always contains the raw text.  skipped=True if the human chose
    to skip or the interaction timed out.
    """

    content: str
    skipped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CallHuman(Protocol):
    """
    Interface: surface a message to a human and collect their response.

    Any callable matching this signature satisfies the protocol.
    This enables:
      - Console input() for CLI drivers
      - Web form handlers for Chainlit / Gradio / Streamlit
      - Mock responses for deterministic testing
      - Slack / Teams / email integrations
    """

    def __call__(self, request: HumanRequest) -> HumanResponse:
        """Present a request to the human and return their response."""
        ...
