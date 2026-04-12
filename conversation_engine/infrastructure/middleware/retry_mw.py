"""
RetryMiddleware — configurable per-node retry with backoff.

Stub implementation: provides the retry infrastructure without
requiring real failure-prone services.  Ready for production use
when real LLM or tool calls are wired in.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Set, Tuple, Type

from conversation_engine.infrastructure.middleware.base import NodeMiddleware

logger = logging.getLogger(__name__)


class RetryMiddleware(NodeMiddleware):
    """
    Retries failed node executions with configurable backoff.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default 2, so 3 total attempts).
    retryable_errors : tuple[Type[Exception], ...]
        Exception types that trigger a retry.
        Default: (Exception,) — retries on any error.
    backoff_base : float
        Base delay in seconds between retries (default 0.1).
        Actual delay = backoff_base * (2 ** attempt) for exponential backoff.
    nodes : set[str] | None
        If provided, only retry these nodes. Critical for targeting
        LLM-calling or tool-calling nodes without retrying pure functions.
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retryable_errors: Tuple[Type[Exception], ...] = (Exception,),
        backoff_base: float = 0.1,
        nodes: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(nodes=nodes)
        self._max_retries = max_retries
        self._retryable_errors = retryable_errors
        self._backoff_base = backoff_base

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff: base * 2^attempt."""
        return self._backoff_base * (2**attempt)

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return next_fn(state)
            except self._retryable_errors as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "[%s] Attempt %d/%d failed (%s), retrying in %.2fs",
                        node_name,
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[%s] All %d attempts exhausted: %s",
                        node_name,
                        self._max_retries + 1,
                        exc,
                    )

        # Should not reach here, but satisfy type checker
        raise last_error  # type: ignore[misc]
