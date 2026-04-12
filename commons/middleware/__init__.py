"""
middleware — Composable cross-cutting concerns for LangGraph nodes.

One ABC, one pattern: each middleware wraps the next, forming a call chain.
The innermost call is the actual node function.

    Logging → Metrics → Validation → Retry → CircuitBreaker → [node]

Each middleware receives (node_name, state, next_fn) and decides:
  - Call next_fn(state) to continue the chain
  - Don't call it — to short-circuit (circuit breaker)
  - Call it multiple times — to retry
  - Transform state before or result after

Per-node selectivity: each middleware carries an optional `nodes` set.
If provided, the middleware only activates for those nodes; otherwise it
applies to all nodes.
"""

from commons.middleware.base import (
    NodeMiddleware,
)
from commons.middleware.circuit_breaker_mw import (
    CircuitBreakerMiddleware,
)
from commons.middleware.config_mw import (
    ConfigMiddleware,
)
from commons.middleware.error_handling_mw import (
    ErrorHandlingMiddleware,
)
from commons.middleware.instrumented_graph import (
    InstrumentedGraph,
)
from commons.middleware.logging_mw import (
    LoggingMiddleware,
)
from commons.middleware.metrics_mw import (
    MetricsMiddleware,
    NodeMetrics,
)
from commons.middleware.retry_mw import (
    RetryMiddleware,
)
from commons.middleware.validation_mw import (
    ValidationMiddleware,
)

__all__ = [
    "NodeMiddleware",
    "InstrumentedGraph",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "NodeMetrics",
    "ValidationMiddleware",
    "ErrorHandlingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "ConfigMiddleware",
]
