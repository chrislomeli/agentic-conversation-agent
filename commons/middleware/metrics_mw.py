"""MetricsMiddleware — thread-safe per-node call counts, durations, errors."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Set

from commons.middleware.base import NodeMiddleware


@dataclass
class NodeMetrics:
    """Accumulated metrics for a single node."""

    call_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    last_duration: float = 0.0


class MetricsMiddleware(NodeMiddleware):
    """
    Collects per-node call counts, error counts, and durations.

    Thread-safe via a threading lock.  Access metrics via
    .metrics dict or .snapshot() for a JSON-safe copy.
    """

    def __init__(self, *, nodes: Optional[Set[str]] = None) -> None:
        super().__init__(nodes=nodes)
        self._lock = threading.Lock()
        self.metrics: dict[str, NodeMetrics] = {}

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        t0 = time.perf_counter()
        try:
            result = next_fn(state)
            elapsed = time.perf_counter() - t0
            with self._lock:
                m = self.metrics.setdefault(node_name, NodeMetrics())
                m.call_count += 1
                m.total_duration += elapsed
                m.last_duration = elapsed
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            with self._lock:
                m = self.metrics.setdefault(node_name, NodeMetrics())
                m.call_count += 1
                m.error_count += 1
                m.total_duration += elapsed
                m.last_duration = elapsed
            raise

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-safe snapshot of all metrics."""
        with self._lock:
            return {
                name: {
                    "call_count": m.call_count,
                    "error_count": m.error_count,
                    "total_duration": round(m.total_duration, 6),
                    "last_duration": round(m.last_duration, 6),
                    "avg_duration": round(m.total_duration / m.call_count, 6)
                    if m.call_count
                    else 0.0,
                }
                for name, m in self.metrics.items()
            }
