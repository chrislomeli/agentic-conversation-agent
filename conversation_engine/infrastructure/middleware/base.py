"""
NodeMiddleware — the single ABC for all cross-cutting concerns.

Design:
  - Each middleware wraps the *entire* node execution via next_fn
  - Middleware can observe, transform, retry, short-circuit, inject config
  - Per-node selectivity via optional `nodes` set
  - Chain is built by InstrumentedGraph at add_node() time
  - Order matters: first in list = outermost wrapper

The signature:
    def __call__(self, node_name: str, state: dict, next_fn: Callable) -> dict

  - Call next_fn(state) to continue the chain (or execute the node)
  - Don't call it to short-circuit
  - Call it multiple times to retry
  - Transform state before or result after
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class NodeMiddleware(ABC):
    """
    Base class for composable node-level cross-cutting concerns.

    Subclass and implement __call__. Use self.applies_to(node_name)
    to check whether this middleware should activate for a given node.

    Parameters
    ----------
    nodes : set[str] | None
        If provided, this middleware only activates for these node names.
        If None, it activates for all nodes.
    """

    def __init__(self, *, nodes: set[str] | None = None) -> None:
        self._nodes: frozenset[str] | None = frozenset(nodes) if nodes else None

    def applies_to(self, node_name: str) -> bool:
        """Return True if this middleware should activate for the given node."""
        return self._nodes is None or node_name in self._nodes

    @abstractmethod
    def __call__(self, node_name: str, state: Any, next_fn: Callable[[Any], Any]) -> Any:
        """
        Execute this middleware's concern around the node.

        Args:
            node_name: The name of the node being executed.
            state: The current graph state dict.
            next_fn: Call this with state to continue the chain.
                     The innermost next_fn is the actual node function.

        Returns:
            The node's result dict (possibly transformed).
        """
        ...
