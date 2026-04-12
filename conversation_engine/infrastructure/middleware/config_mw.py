"""
ConfigMiddleware — injects per-node configuration into state.

Allows each node to receive custom configuration without coupling
the node implementation to a global config system.  The config dict
is injected as state["_node_config"] before the node runs.

Stub implementation: ready for real config sources (files, env, remote).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from conversation_engine.infrastructure.middleware.base import NodeMiddleware

logger = logging.getLogger(__name__)


class ConfigMiddleware(NodeMiddleware):
    """
    Injects per-node configuration into the state dict.

    Parameters
    ----------
    config : dict[str, dict[str, Any]]
        Mapping of node_name → config dict.
        Nodes not in this dict receive an empty config.
    config_key : str
        The state key to inject config under (default "_node_config").
    nodes : set[str] | None
        Optional node filter.
    """

    def __init__(
        self,
        config: dict[str, dict[str, Any]],
        *,
        config_key: str = "_node_config",
        nodes: set[str] | None = None,
    ) -> None:
        super().__init__(nodes=nodes)
        self._config = config
        self._config_key = config_key

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        node_config = self._config.get(node_name, {})
        # Inject config into a copy of state so the original isn't mutated
        augmented = {**state, self._config_key: node_config}
        return next_fn(augmented)
