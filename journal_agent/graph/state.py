from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from journal_agent.model.session import ClassifiedExchange, Fragment

STATUS_IDLE = "idle"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"


class JournalState(TypedDict):
    session_id: str
    seed_context: list[BaseMessage]
    session_messages: Annotated[list[BaseMessage], add_messages]
    classified_exchanges: list[ClassifiedExchange]  # new — written by classify, read by extract
    fragments: list[Fragment]  # new — written by classify, read by extract
    status: Literal["idle", "processing", "completed", "error"]
    error_message: str | None
