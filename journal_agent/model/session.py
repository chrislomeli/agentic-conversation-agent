from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Role(Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    NONE = "none"

class ClassifiedExchange(BaseModel):
    id: str                    # uuid
    session_id: str            # back-reference to raw session
    turn_indices: tuple[int, int]  # human turn idx, ai turn idx
    human_summary: str         # condensed question
    ai_summary: str            # condensed answer
    subject: str               # from SUBJECT_TAXONOMY
    themes: list[str]          # from THEME_TAXONOMY
    timestamp: datetime

class Turn(BaseModel):
    session_id: str
    role: Role
    content: str
    metadata: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

class Fragment(BaseModel):
    id: str
    content: str
    tags: list[str]
    summary: str
    timestamp: datetime
    session_id: str  # (which Turn this came from)

