"""
CircuitBreakerMiddleware — prevents repeated calls to failing nodes.

Three states:
  CLOSED   — normal operation, calls pass through
  OPEN     — node is failing, calls are short-circuited
  HALF_OPEN — after cooldown, one probe call is allowed through
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from commons.middleware.base import NodeMiddleware
from commons.node_validation.result_schema import NodeResult

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _CircuitStatus:
    """Per-node circuit breaker state."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0


class CircuitBreakerMiddleware(NodeMiddleware):
    """
    Prevents cascading failures by short-circuiting calls to failing nodes.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before opening the circuit (default 3).
    cooldown_seconds : float
        Time to wait before allowing a probe call (default 30.0).
    success_threshold : int
        Number of consecutive successes in half-open state to close
        the circuit (default 1).
    nodes : set[str] | None
        If provided, only these nodes get circuit-breaker protection.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        success_threshold: int = 1,
        nodes: set[str] | None = None,
    ) -> None:
        super().__init__(nodes=nodes)
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._success_threshold = success_threshold
        self._lock = threading.Lock()
        self._circuits: dict[str, _CircuitStatus] = {}

    def _get_circuit(self, node_name: str) -> _CircuitStatus:
        if node_name not in self._circuits:
            self._circuits[node_name] = _CircuitStatus()
        return self._circuits[node_name]

    def get_state(self, node_name: str) -> CircuitState:
        """Return the current circuit state for a node (for testing/monitoring)."""
        with self._lock:
            return self._get_circuit(node_name).state

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        with self._lock:
            circuit = self._get_circuit(node_name)

            if circuit.state == CircuitState.OPEN:
                # Check if cooldown has elapsed
                if time.monotonic() - circuit.last_failure_time >= self._cooldown:
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.success_count = 0
                    logger.info("[%s] Circuit half-open, allowing probe call", node_name)
                else:
                    logger.warning("[%s] Circuit OPEN, short-circuiting", node_name)
                    return {
                        "node_result": NodeResult.failure(
                            code="CIRCUIT_OPEN",
                            message=f"Circuit breaker open for node '{node_name}'",
                            details={"failure_count": circuit.failure_count},
                        ),
                        "status": "error",
                    }

        # Execute the node (outside the lock)
        try:
            result = next_fn(state)
        except Exception:
            with self._lock:
                circuit = self._get_circuit(node_name)
                circuit.failure_count += 1
                circuit.last_failure_time = time.monotonic()
                if circuit.failure_count >= self._failure_threshold:
                    circuit.state = CircuitState.OPEN
                    logger.error(
                        "[%s] Circuit OPENED after %d failures",
                        node_name,
                        circuit.failure_count,
                    )
            raise

        # Success
        with self._lock:
            circuit = self._get_circuit(node_name)
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.success_count += 1
                if circuit.success_count >= self._success_threshold:
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    logger.info("[%s] Circuit CLOSED after successful probe", node_name)
            else:
                circuit.failure_count = 0  # Reset on success

        return result
