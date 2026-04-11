from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Role(Enum):
    HUMAN = "human"
    AI = "ai"
    NONE = "none"

class Turn(BaseModel):
    exchange_id: str
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

