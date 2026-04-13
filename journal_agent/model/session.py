import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Role(Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    NONE = "none"


@dataclass
class Ideation:
    tag: str
    goals: str
    example: str


class Turn(BaseModel):
    session_id: str
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Tag(BaseModel):
    tag: str
    note: str | None = None


class Fragment(BaseModel):
    id: str  # a unique uuid
    session_id: str  # (which Turn this came from)
    content: str  # your summary for searchable embedding
    exchange_ids: list[str]  # a list of exchange_id from the Exchange records that comprised this summary
    tags: list[Tag]  # tags
    timestamp: datetime


class Exchange(BaseModel):
    exchange_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None  # back-reference to raw session
    human: Turn | None = None
    ai: Turn | None = None


class ClassifiedExchange(BaseModel):
    session_id: str  # copied from the Exchange record
    exchange_id: list[str]  # a list of exchange_id from the Exchange records that comprised this summary
    human_summary: str  # transcribe the relevant parts of the human message or copy it if it does not need condensing
    ai_summary: str  # transcribe the relevant parts of the AI message or copy it if it does not need condensing
    tags: list[Tag]  # from TAXONOMY classify this record using the TAXONOMY provided
    timestamp: datetime  # copied from the Exchange record
