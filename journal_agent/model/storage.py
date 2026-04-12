from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Role(Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    NONE = "none"


class Turn(BaseModel):
    session_id: str
    role: Role
    content: str
    metadata: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
