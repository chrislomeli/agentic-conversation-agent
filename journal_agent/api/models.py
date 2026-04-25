"""API request and response models.

These cross the HTTP boundary — kept separate from JournalState so the API
contract can evolve independently of the graph.
"""
from enum import StrEnum

from pydantic import BaseModel


class MessageRole(StrEnum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class ChatRequest(BaseModel):
    message: str


class SessionResponse(BaseModel):
    session_id: str


class SseEvent(StrEnum):
    """SSE event type names sent in the ``event:`` field."""

    TOKEN = "token"   # one chunk of AI text
    SYSTEM = "system" # feedback from the graph (e.g. /save confirmation)
    DONE = "done"     # stream complete
    ERROR = "error"   # something went wrong
