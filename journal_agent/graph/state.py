from typing import Annotated, Literal, TypedDict
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from journal_agent.model.session import ClassifiedExchange, Fragment, Exchange, ThreadSegment, ExpandedThreadSegment, \
    ScoreCard, ContextSpecification

STATUS_IDLE = "idle"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_TRANSCRIPT_SAVED = "transcript_saved"
STATUS_EXCHANGES_SAVED = "exchanges_saved"   # deprecated — old one-shot pipeline
STATUS_THREADS_SAVED = "threads_saved"
STATUS_CLASSIFIED_THREADS_SAVED = "classified_threads_saved"
STATUS_FRAGMENTS_SAVED = "fragments_saved"
STATUS_ERROR = "error"


class JournalState(TypedDict):
    session_id: str
    recent_messages: list[BaseMessage]
    session_messages: Annotated[list[BaseMessage], add_messages]
    transcript: Annotated[list[Exchange], add]
    threads: Annotated[list[ThreadSegment], add]
    classified_threads: Annotated[list[ThreadSegment], add]
    fragments: list[Fragment]  # new — written by classify, read by extract
    retrieved_history: list[Fragment]  # lookup by user query
    context_specification: ContextSpecification  # current instruction from intent classification
    status: Literal["idle", "processing", "completed", "error"]
    error_message: str | None
