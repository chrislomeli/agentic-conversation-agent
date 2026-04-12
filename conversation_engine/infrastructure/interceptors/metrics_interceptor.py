"""
MetricsInterceptor — Collects per-node call counts, durations, and error counts.

Thread-safe: uses a threading lock so it works with parallel Send() fan-outs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from conversation_engine.infrastructure.instrumented_graph import Interceptor


@dataclass
class NodeMetrics:
    call_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    last_duration: float = 0.0


class MetricsInterceptor(Interceptor):
    """
    Accumulates lightweight metrics per node.

    Access via .metrics dict (keyed by node_name) or .snapshot().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_times: dict[str, float] = {}
        self.metrics: dict[str, NodeMetrics] = {}

    def before(self, node_name: str, state: Any) -> None:
        self._start_times[node_name] = time.perf_counter()

    def after(self, node_name: str, state: Any, result: Any) -> None:
        elapsed = time.perf_counter() - self._start_times.pop(node_name, 0.0)
        with self._lock:
            m = self.metrics.setdefault(node_name, NodeMetrics())
            m.call_count += 1
            m.total_duration += elapsed
            m.last_duration = elapsed

    def on_error(self, node_name: str, state: Any, error: Exception) -> None:
        elapsed = time.perf_counter() - self._start_times.pop(node_name, 0.0)
        with self._lock:
            m = self.metrics.setdefault(node_name, NodeMetrics())
            m.call_count += 1
            m.error_count += 1
            m.total_duration += elapsed
            m.last_duration = elapsed

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
