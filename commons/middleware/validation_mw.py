"""
ValidationMiddleware — validates node input against Pydantic schemas.

Replaces the former @validated_node decorator with a composable middleware
that can be configured per-node via a schemas dict.

If validation fails, returns a NodeResult.failure without calling the node.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set, Type

from pydantic import BaseModel, ValidationError

from commons.middleware.base import NodeMiddleware
from commons.node_validation.result_schema import NodeResult

logger = logging.getLogger(__name__)


class ValidationMiddleware(NodeMiddleware):
    """
    Validates state against a Pydantic schema before each node runs.

    Parameters
    ----------
    schemas : dict[str, Type[BaseModel]]
        Mapping of node_name → Pydantic model to validate against.
        Nodes not in this dict are passed through without validation.
    nodes : set[str] | None
        Optional node filter (inherits from NodeMiddleware).
        If provided, only these nodes are validated even if they
        appear in schemas.  Usually you just use schemas and leave
        nodes=None.
    """

    def __init__(
        self,
        schemas: Dict[str, Type[BaseModel]],
        *,
        nodes: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(nodes=nodes)
        self._schemas = schemas

    def __call__(self, node_name: str, state: Any, next_fn: Callable) -> Any:
        if not self.applies_to(node_name):
            return next_fn(state)

        schema = self._schemas.get(node_name)
        if schema is None:
            return next_fn(state)

        try:
            schema.model_validate(state)
        except ValidationError as e:
            logger.warning(
                "[%s] Input validation failed: %s",
                node_name,
                e.error_count(),
            )
            return {
                "node_result": NodeResult.failure(
                    code="INVALID_INPUT",
                    message=f"{schema.__name__} validation failed for node '{node_name}'",
                    details={"errors": e.errors()},
                )
            }

        return next_fn(state)
