"""LoggingMiddleware — structured entry/exit/error logging with timing."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from conversation_engine.infrastructure.middleware.base import NodeMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(NodeMiddleware):
    """
    Logs node entry, exit (with duration), and errors.

    This is the outermost middleware in a typical chain so that
    timing captures everything including retry/circuit-breaker overhead.
    """

    def __init__(
        self,
        *,
        nodes: set[str] | None = None,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(nodes=nodes)
        self._level = level

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        logger.log(self._level, "[%s] ▶ entering", node_name)
        t0 = time.perf_counter()
        try:
            result = next_fn(state)
            elapsed = time.perf_counter() - t0
            logger.log(self._level, "[%s] ✓ completed (%.3fs)", node_name, elapsed)
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("[%s] ✗ failed after %.3fs: %s", node_name, elapsed, e)
            raise
