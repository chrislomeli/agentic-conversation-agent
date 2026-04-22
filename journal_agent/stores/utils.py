"""utils.py — Shared helpers for the stores layer."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from journal_agent.model.session import Exchange, Role


def exchanges_to_messages(exchanges: list[Exchange]) -> list[BaseMessage]:
    """Convert Exchange records to a flat list of LangChain messages.

    Ordering: for each exchange, human turn first, then AI turn.
    """
    messages: list[BaseMessage] = []
    for exchange in exchanges:
        if exchange.human:
            if exchange.human.role == Role.HUMAN:
                messages.append(HumanMessage(content=exchange.human.content))
            elif exchange.human.role == Role.SYSTEM:
                messages.append(SystemMessage(content=exchange.human.content))
        if exchange.ai:
            messages.append(AIMessage(content=exchange.ai.content))
    return messages


def resolve_project_root() -> Path:
    """Determine the stores root directory.

    Resolution order:
      1. JOURNAL_AGENT_ROOT env var (explicit override, used by tests)
      2. Walk up from this file until we find pyproject.toml
      3. Fall back to cwd
    """
    configured_root = os.getenv("JOURNAL_AGENT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    return Path.cwd().resolve()