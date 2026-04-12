"""
node_validation — Structured result envelope + input validation for LangGraph nodes.

- NodeResult — structured ok/error envelope every node can return
- validated_node — decorator that validates state against a Pydantic schema
- handle_error — generic error-handler node that reads NodeResult and marks failure
"""

from commons.node_validation.handle_error import (
    handle_error,
)
from commons.node_validation.result_schema import (
    NodeError,
    NodeResult,
)
from commons.node_validation.validator_decorator import (
    validated_node,
)

__all__ = [
    "NodeError",
    "NodeResult",
    "validated_node",
    "handle_error",
]
