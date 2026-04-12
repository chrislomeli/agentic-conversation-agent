"""
InstrumentedGraph -- StateGraph subclass that intercepts add_node()
to wrap every node function with a composable NodeMiddleware chain.

Each middleware wraps the entire execution — it can observe, transform,
retry, short-circuit, or inject config.  The chain is built once per
node at add_node() time and follows the pattern:

    Logging → Metrics → Validation → Retry → CircuitBreaker → [node]

Design principles:
  - Single ABC (NodeMiddleware) replaces the former Interceptor + Middleware split
  - functools.wraps preserves __name__, __doc__, __module__, __qualname__, __wrapped__
  - Per-node selectivity: each middleware decides via applies_to(node_name)
  - Chain is built inside-out: last middleware in list is closest to the node
  - Forward-compatible add_node(**kwargs) -- passes through any future LangGraph parameters
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from langgraph.graph import StateGraph

from conversation_engine.infrastructure.middleware.base import NodeMiddleware

logger = logging.getLogger(__name__)


# -- Legacy ABCs (deprecated, kept for transition) --------------------
# These will be removed once all code migrates to NodeMiddleware.


class Interceptor(ABC):
    """DEPRECATED: Use NodeMiddleware instead."""

    @abstractmethod
    def before(self, node_name: str, state: Any) -> None: ...

    @abstractmethod
    def after(self, node_name: str, state: Any, result: Any) -> None: ...

    @abstractmethod
    def on_error(self, node_name: str, state: Any, error: Exception) -> None: ...


class Middleware(ABC):
    """DEPRECATED: Use NodeMiddleware instead."""

    @abstractmethod
    def transform(self, node_name: str, state: Any, result: Any) -> Any: ...


# -- InstrumentedGraph ------------------------------------------------


class InstrumentedGraph(StateGraph):
    """
    Drop-in replacement for StateGraph that wraps every node with
    a composable NodeMiddleware chain.

    Usage:
        from conversation_engine.infrastructure.middleware import (
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
        # Legacy parameters — deprecated, kept for backwards compatibility
        interceptors: Sequence[Interceptor] | None = None,
        middleware: Sequence[Middleware] | None = None,
        **kwargs: Any,
    ):
        super().__init__(state_schema, **kwargs)
        self._node_middleware: list[NodeMiddleware] = list(node_middleware or [])
        # Legacy support
        self._interceptors: list[Interceptor] = list(interceptors or [])
        self._middleware: list[Middleware] = list(middleware or [])

    # -- public API ---------------------------------------------------

    def add_node_middleware(self, mw: NodeMiddleware) -> None:
        """Add a NodeMiddleware after construction."""
        self._node_middleware.append(mw)

    # Legacy methods — deprecated
    def add_interceptor(self, interceptor: Interceptor) -> None:
        """DEPRECATED: Use add_node_middleware() instead."""
        self._interceptors.append(interceptor)

    def add_middleware(self, mw: Middleware) -> None:
        """DEPRECATED: Use add_node_middleware() instead."""
        self._middleware.append(mw)

    def add_node(self, node: str, action: Callable | None = None, **kwargs: Any) -> None:
        """Override: wrap *action* with middleware chain, then delegate to super()."""
        if action is not None:
            has_new = bool(self._node_middleware)
            has_legacy = bool(self._interceptors or self._middleware)
            if has_new:
                action = self._wrap_chain(node, action)
            elif has_legacy:
                action = self._wrap_legacy(node, action)
        super().add_node(node, action, **kwargs)

    # -- new middleware chain -----------------------------------------

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

    # -- legacy wrapping (deprecated) ---------------------------------

    def _wrap_legacy(self, node_name: str, fn: Callable) -> Callable:
        """Legacy wrapper for old-style Interceptor + Middleware."""
        interceptors = self._interceptors
        middleware = self._middleware

        @functools.wraps(fn)
        def wrapper(state: Any, **kwargs: Any) -> Any:
            for ic in interceptors:
                try:
                    ic.before(node_name, state)
                except Exception:
                    logger.exception(
                        "Interceptor %s.before() failed for node '%s'", type(ic).__name__, node_name
                    )

            try:
                result = fn(state, **kwargs)
            except Exception as exc:
                for ic in interceptors:
                    try:
                        ic.on_error(node_name, state, exc)
                    except Exception:
                        logger.exception(
                            "Interceptor %s.on_error() failed for node '%s'",
                            type(ic).__name__,
                            node_name,
                        )
                raise

            for ic in interceptors:
                try:
                    ic.after(node_name, state, result)
                except Exception:
                    logger.exception(
                        "Interceptor %s.after() failed for node '%s'", type(ic).__name__, node_name
                    )

            for mw in middleware:
                result = mw.transform(node_name, state, result)

            return result

        return wrapper
