"""state.py — LangGraph state definition for the journal agent.

Fields annotated with ``add`` or ``add_messages`` are *append-reducers*:
each node returns a partial dict and LangGraph merges it into the
accumulated state using the annotated reducer function.
"""
from datetime import datetime
from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from journal_agent.model.insights import (
    ProposedSubject,
    Subject,
    Vote,
)
from journal_agent.model.session import (
    Cluster,
    ContextSpecification,
    Exchange,
    Fragment,
    Insight,
    StatusValue,
    ThreadSegment,
    UserCommandValue,
    UserProfile,
)


class WindowParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    window_start: datetime | None = None
    window_end: datetime | None = None
    limit: int | None = None


class ReflectionState(BaseModel):
    session_id: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fetch_parameters: WindowParams | None = None
    fragments: list[Fragment] = Field(default_factory=list)

    # Phase 10 (cluster-based) — kept for coexistence with the old reflection graph.
    clusters: list[Cluster] = Field(default_factory=list)
    uncategorized_fragments: list[str] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    verified_insights: list[Insight] = Field(default_factory=list)
    latest_insights: list[Insight] = Field(default_factory=list)

    # Phase 11 (claim-based) — populated by the new reflection graph.
    candidate_subjects: list[Subject] = Field(
        default_factory=list,
        description="Subjects routed as candidates for the current fragment(s) by route_candidates.",
    )
    votes: list[Vote] = Field(
        default_factory=list,
        description="Votes produced by classify_stance, ready to persist.",
    )
    proposed_subject: ProposedSubject | None = Field(
        default=None,
        description="A new subject proposed by the proposer node, if any.",
    )

    status: StatusValue = StatusValue.IDLE
    error_message: str | None = None


class JournalState(BaseModel):
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
        system_message        — feedback string for the user; printed by the run loop, not by nodes
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    recent_messages: list[BaseMessage] = Field(default_factory=list)
    session_messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    transcript: Annotated[list[Exchange], add] = Field(default_factory=list)
    threads: Annotated[list[ThreadSegment], add] = Field(default_factory=list)
    classified_threads: Annotated[list[ThreadSegment], add] = Field(default_factory=list)
    fragments: list[Fragment] = Field(default_factory=list)
    retrieved_history: list[Fragment] = Field(default_factory=list)
    fetch_parameters: WindowParams | None = None
    latest_insights: list[Insight] = Field(default_factory=list)
    context_specification: ContextSpecification = Field(default_factory=ContextSpecification)
    user_profile: UserProfile = Field(default_factory=UserProfile)
    user_command: UserCommandValue = UserCommandValue.NONE
    user_command_args: str | None = None
    status: StatusValue = StatusValue.IDLE
    error_message: str | None = None
    system_message: str | None = None
