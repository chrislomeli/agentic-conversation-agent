"""
ErrorHandlingMiddleware — catches exceptions and NodeResult failures.

Replaces the former handle_error node + conditional routing topology.
Instead of routing to a separate node on failure, this middleware
catches errors inline and sets status="error" in the result.

Two modes of error catching:
  1. Exception from the node → caught, logged, returns NodeResult.failure
  2. NodeResult.failure in the result dict → sets status="error"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from conversation_engine.infrastructure.middleware.base import NodeMiddleware
from conversation_engine.infrastructure.node_validation.result_schema import NodeResult

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(NodeMiddleware):
    """
    Catches node exceptions and NodeResult failures.

    When a node raises:
      - Logs the exception
      - Returns {"node_result": NodeResult.failure(...), "status": "error"}

    When a node returns a NodeResult with ok=False:
      - Logs the failure
      - Merges {"status": "error"} into the result

    Parameters
    ----------
    swallow_exceptions : bool
        If True (default), exceptions are caught and converted to
        NodeResult.failure. If False, exceptions propagate after
        being logged.
    nodes : set[str] | None
        Optional node filter.
    """

    def __init__(
        self,
        *,
        nodes: set[str] | None = None,
        swallow_exceptions: bool = True,
    ) -> None:
        super().__init__(nodes=nodes)
        self._swallow = swallow_exceptions

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        try:
            result = next_fn(state)
        except Exception as exc:
            logger.error("[%s] Exception caught: %s", node_name, exc)
            if self._swallow:
                return {
                    "node_result": NodeResult.failure(
                        code="NODE_EXCEPTION",
                        message=f"Node '{node_name}' raised: {exc}",
                        details={"exception_type": type(exc).__name__},
                    ),
                    "status": "error",
                }
            raise

        # Check for NodeResult failure in the result
        if isinstance(result, dict):
            node_result = result.get("node_result")
            if isinstance(node_result, NodeResult) and not node_result.ok:
                logger.error(
                    "[%s] Node returned failure: [%s] %s",
                    node_name,
                    node_result.error.code if node_result.error else "UNKNOWN",
                    node_result.error.message if node_result.error else "",
                )
                result["status"] = "error"

        return result
