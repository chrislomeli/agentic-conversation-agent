"""state.py — LangGraph state definition for the journal agent.

JournalState is a TypedDict that every node reads from and writes to.
Fields annotated with ``add`` or ``add_messages`` are *append-reducers*:
each node returns a partial dict and LangGraph merges it into the
accumulated state using the annotated reducer function.
"""
from datetime import datetime
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from journal_agent.model.session import Fragment, Exchange, ThreadSegment, ContextSpecification, Status, UserProfile, \
    Cluster, Insight


class ReflectionState(TypedDict):
    window_start: datetime
    window_end: datetime
    fragments: list[Fragment]            # populated by collect_window
    clusters: list[Cluster]              # populated (and scored) by cluster_fragments
    insights: list[Insight]              # populated by label_clusters
    # verified_insights: list[Insight]     # populated by verify_citations
    summary: str                         # populated by summarize_result
    error: str | None


class JournalState(TypedDict):
    """Shared state flowing through all graph nodes.

    Conversation loop (runs every turn):
        session_id            — UUID for the current session
        recent_messages       — messages from the *previous* session (seed context)
        session_messages      — accumulates Human/AI messages in this session (append-reducer)
        context_specification — set by intent_classifier; drives prompt + retrieval config
        retrieved_history     — Fragments from vector search, used to enrich the system prompt

    End-of-session pipeline (runs once after /quit):
        transcript            — completed Exchange pairs, appended each turn (append-reducer)
        threads               — ThreadSegments from the exchange decomposer (append-reducer)
        classified_threads    — ThreadSegments with taxonomy tags (append-reducer)
        fragments             — standalone ideas extracted from classified threads

    Control flow:
        status                — routing signal: IDLE → PROCESSING → COMPLETED / ERROR
        error_message         — set alongside Status.ERROR to propagate failure info
    """
    session_id: str
    recent_messages: list[BaseMessage]
    session_messages: Annotated[list[BaseMessage], add_messages]
    transcript: Annotated[list[Exchange], add]
    threads: Annotated[list[ThreadSegment], add]
    classified_threads: Annotated[list[ThreadSegment], add]
    fragments: list[Fragment]
    retrieved_history: list[Fragment]
    context_specification: ContextSpecification
    user_profile: UserProfile
    status: Status
    error_message: str | None
