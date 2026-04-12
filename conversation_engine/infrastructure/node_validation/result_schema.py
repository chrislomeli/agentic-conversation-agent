"""Structured result envelope for LangGraph nodes."""

from typing import Any

from pydantic import BaseModel, Field


class NodeError(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class NodeResult(BaseModel):
    ok: bool
    data: Any | None = None
    error: NodeError | None = None

    @classmethod
    def success(cls, data: Any = None) -> "NodeResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, code: str, message: str, details: dict | None = None) -> "NodeResult":
        return cls(ok=False, error=NodeError(code=code, message=message, details=details or {}))
