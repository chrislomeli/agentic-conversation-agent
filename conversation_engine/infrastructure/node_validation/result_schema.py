"""Structured result envelope for LangGraph nodes."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class NodeError(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class NodeResult(BaseModel):
    ok: bool
    data: Optional[Any] = None
    error: Optional[NodeError] = None

    @classmethod
    def success(cls, data: Any = None) -> "NodeResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, code: str, message: str, details: Optional[dict] = None) -> "NodeResult":
        return cls(ok=False, error=NodeError(code=code, message=message, details=details or {}))
