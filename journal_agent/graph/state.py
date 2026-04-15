from typing import Annotated, Literal, TypedDict
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from journal_agent.model.session import ClassifiedExchange, Fragment, Exchange, ThreadSegment, ExpandedThreadSegment

STATUS_IDLE = "idle"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_TRANSCRIPT_SAVED = "transcript_saved"
STATUS_EXCHANGES_SAVED = "exchanges_saved"
STATUS_FRAGMENTS_SAVED = "fragments_saved"
STATUS_ERROR = "error"


class JournalState(TypedDict):
    session_id: str
    seed_context: list[BaseMessage]
    session_messages: Annotated[list[BaseMessage], add_messages]
    transcript: Annotated[list[Exchange], add]
    threads: Annotated[list[ThreadSegment], add]
    classified_threads: Annotated[list[ThreadSegment], add]
    classified_exchanges: list[ClassifiedExchange]  # todo DEPRECATEm THIS
    fragments: list[Fragment]  # new — written by classify, read by extract
    status: Literal["idle", "processing", "completed", "error"]
    error_message: str | None
