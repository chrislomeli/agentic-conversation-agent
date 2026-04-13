"""
llm_registry.py — Named LLM client catalog.

Holds a mapping of role-name → LLMClient so that any node can request the
client it needs by key (e.g. "conversation", "classifier") without knowing
how it was constructed.

Usage::

    registry = build_llm_registry(settings)
    conversation_llm = registry.get("conversation")
    classifier_llm   = registry.get("classifier")

The registry is built once at startup (in main.py) and threaded into the
graph builder.  Adding a new role is two steps:
  1. Add an entry to ``LLM_ROLE_CONFIG`` in ``configure/config_builder.py``.
  2. Use ``registry.get("new_role")`` in the node factory that needs it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from journal_agent.comms.llm_client import LLMClient, create_llm_client
from journal_agent.configure.settings import LLMLabel, LLMModel, Settings

logger = logging.getLogger(__name__)


# ── Registry ──────────────────────────────────────────────────────────────

@dataclass
class LLMRegistry:
    """Immutable catalog of named LLMClient instances."""

    _clients: dict[str, LLMClient] = field(default_factory=dict)

    # -- public API --------------------------------------------------------

    def get(self, role: str) -> LLMClient:
        """Return the client for *role*, falling back to ``"conversation"``."""
        client = self._clients.get(role)
        if client is not None:
            return client
        fallback = self._clients.get("conversation")
        if fallback is not None:
            logger.warning(
                "No LLM configured for role %r — falling back to 'conversation'",
                role,
            )
            return fallback
        raise KeyError(
            f"No LLM registered for role {role!r} and no 'conversation' fallback."
        )

    @property
    def roles(self) -> list[str]:
        return sorted(self._clients)


# ── Builder ───────────────────────────────────────────────────────────────

def _resolve_model(
    role: str,
    role_config: dict[str, LLMLabel],
    models: dict[LLMLabel, LLMModel | None],
    settings: Settings,
) -> LLMModel | None:
    """Look up the LLMModel for a role, injecting the API key from settings."""
    label = role_config.get(role)
    if label is None:
        return None
    model_cfg = models.get(label)
    if model_cfg is None:
        return None
    # Attach the live API key from settings (same logic as Settings.selected_model)
    import dataclasses as _dc

    resolved = _dc.replace(model_cfg)
    raw_secret = getattr(settings, resolved.key_label, None)
    resolved.api_key = raw_secret or None
    return resolved


def build_llm_registry(
    settings: Settings,
    models: dict[LLMLabel, LLMModel | None],
    role_config: dict[str, LLMLabel],
) -> LLMRegistry:
    """
    Construct an LLMRegistry from application settings.

    Parameters
    ----------
    settings:
        The loaded Settings (carries API keys and ollama_base_url).
    models:
        The label→LLMModel mapping from config_builder.
    role_config:
        Role-name → LLMLabel mapping (defined in configure/config_builder.py).
    """
    effective_config = role_config
    clients: dict[str, LLMClient] = {}

    for role, label in effective_config.items():
        model_cfg = _resolve_model(role, effective_config, models, settings)
        if model_cfg is None:
            logger.warning("Skipping role %r — no model found for label %r", role, label)
            continue
        client = create_llm_client(
            provider=model_cfg.provider,
            api_key=model_cfg.api_key,
            model=model_cfg.model,
            base_url=settings.ollama_base_url,
        )
        logger.info("Registered LLM for role %r → %s (%s)", role, model_cfg.model, model_cfg.provider.value)
        clients[role] = client

    return LLMRegistry(_clients=clients)
