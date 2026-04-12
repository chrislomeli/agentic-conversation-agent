"""
node_validation — Backwards-compatibility shim.

Reusable node validation has moved to ``commons.node_validation``.
"""

from commons.node_validation import (
    NodeError,
    NodeResult,
    handle_error,
    validated_node,
)

__all__ = [
    "NodeError",
    "NodeResult",
    "validated_node",
    "handle_error",
]
