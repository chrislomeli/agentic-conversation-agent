"""
InstrumentedGraph -- StateGraph subclass that intercepts add_node()
to wrap every node function with a composable NodeMiddleware chain.

Each middleware wraps the entire execution — it can observe, transform,
retry, short-circuit, or inject config.  The chain is built once per
node at add_node() time and follows the pattern:

    Logging → Metrics → Validation → Retry → CircuitBreaker → [node]

Design principles:
  - Single ABC (NodeMiddleware) — no legacy Interceptor/Middleware split
  - functools.wraps preserves __name__, __doc__, __module__, __qualname__, __wrapped__
  - Per-node selectivity: each middleware decides via applies_to(node_name)
  - Chain is built inside-out: last middleware in list is closest to the node
  - Forward-compatible add_node(**kwargs) -- passes through any future LangGraph parameters
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from typing import Any

from langgraph.graph import StateGraph

from commons.middleware.base import NodeMiddleware

logger = logging.getLogger(__name__)


class InstrumentedGraph(StateGraph):
    """
    Drop-in replacement for StateGraph that wraps every node with
    a composable NodeMiddleware chain.

    Usage:
        from commons.middleware import (
            InstrumentedGraph,
            LoggingMiddleware, MetricsMiddleware, RetryMiddleware,
        )

        graph = InstrumentedGraph(
            MyState,
            node_middleware=[
                LoggingMiddleware(),
                MetricsMiddleware(),
                RetryMiddleware(nodes={"reason"}),
            ],
        )
        graph.add_node("my_node", my_func)   # automatically instrumented
    """

    def __init__(
        self,
        state_schema: Any,
        *,
        node_middleware: Sequence[NodeMiddleware] | None = None,
        **kwargs: Any,
    ):
        super().__init__(state_schema, **kwargs)
        self._node_middleware: list[NodeMiddleware] = list(node_middleware or [])

    def add_node_middleware(self, mw: NodeMiddleware) -> None:
        """Add a NodeMiddleware after construction."""
        self._node_middleware.append(mw)

    def add_node(self, node: str, action: Callable | None = None, **kwargs: Any) -> None:
        """Override: wrap *action* with middleware chain, then delegate to super()."""
        if action is not None and self._node_middleware:
            action = self._wrap_chain(node, action)
        super().add_node(node, action, **kwargs)

    def _wrap_chain(self, node_name: str, fn: Callable) -> Callable:
        """Build a next_fn chain from the NodeMiddleware list."""
        mw_list = self._node_middleware

        @functools.wraps(fn)
        def wrapper(state: Any, **kwargs: Any) -> Any:
            # Build the chain inside-out:
            # mw_list[0] wraps mw_list[1] wraps ... wraps fn
            def make_next(index: int) -> Callable:
                if index >= len(mw_list):
                    # Innermost: call the actual node function, forwarding kwargs (e.g. config)
                    def leaf(s: Any) -> Any:
                        return fn(s, **kwargs)

                    return leaf

                mw = mw_list[index]
                inner = make_next(index + 1)

                def chain_step(s: Any) -> Any:
                    return mw(node_name, s, inner)

                return chain_step

            chain = make_next(0)
            return chain(state)

        return wrapper
