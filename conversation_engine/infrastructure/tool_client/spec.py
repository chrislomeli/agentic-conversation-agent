"""
ToolSpec — Canonical tool contract.

Each tool has a name, description, Pydantic input/output models, and a handler.
The Pydantic models serve as the single source of truth for schema validation
and can be exported to JSON Schema for MCP compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolSpec:
    """Immutable definition of a tool's contract."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the input model (MCP-compatible)."""
        return self.input_model.model_json_schema()

    def output_schema(self) -> dict[str, Any]:
        """JSON Schema for the output model (MCP-compatible)."""
        return self.output_model.model_json_schema()
