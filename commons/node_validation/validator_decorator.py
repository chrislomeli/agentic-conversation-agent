"""Decorator that validates LangGraph node input against a Pydantic schema."""

import functools
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from commons.node_validation.result_schema import NodeResult

T = TypeVar("T", bound=BaseModel)


def validated_node(schema: type[T]):
    """
    Decorator that:
      1. Validates state against the node's input schema
      2. Passes a typed input object to the node function
      3. Returns a structured error into state on failure — never throws

    The schema should use ``model_config = {"extra": "ignore"}`` so that
    extra state keys are silently dropped during validation.

    Node signature changes from:
        def my_node(state: dict) -> dict
    to:
        def my_node(inp: MySchema, state: dict) -> dict
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            try:
                inp = schema.model_validate(state)
            except ValidationError as e:
                return {
                    "node_result": NodeResult.failure(
                        code="INVALID_INPUT",
                        message=f"{schema.__name__} validation failed",
                        details={"errors": e.errors()},
                    )
                }
            return fn(inp, state)

        return wrapper

    return decorator
