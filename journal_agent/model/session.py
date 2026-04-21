"""session.py — Domain models for the journal agent pipeline.

Defines every data structure that flows through the LangGraph:

    Turn → Exchange → ThreadSegment → ExpandedThreadSegment → Fragment
            │                                                    │
            └──  raw conversation pairs                          └──  searchable embeddings
                 stored as JSONL                                      stored in ChromaDB

Classification models (ScoreCard, Domain, ContextSpecification) drive
per-turn routing: which prompt to use, how much history to retrieve, etc.

UserProfile (Phase 9) accumulates user preferences across sessions.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Literal, Annotated, Union

from pydantic import BaseModel, Field

from journal_agent.configure.config_builder import (
    DEFAULT_RECENT_MESSAGES_COUNT,
    DEFAULT_SESSION_MESSAGES_COUNT,
    DEFAULT_RETRIEVED_HISTORY_COUNT,
    DEFAULT_RETRIEVED_HISTORY_DISTANCE,
    DEFAULT_RESPONSE_STYLE,
    DEFAULT_EXPLANATION_DEPTH,
    DEFAULT_TONE,
    DEFAULT_LEARNING_STYLE,
    DEFAULT_INTERESTS,
    DEFAULT_PET_PEEVES,
    HUMAN_NAME,
    AI_NAME,
)


class Status(StrEnum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    TRANSCRIPT_SAVED = "transcript_saved"
    EXCHANGES_SAVED = "exchanges_saved"
    THREADS_SAVED = "threads_saved"
    CLASSIFIED_THREADS_SAVED = "classified_threads_saved"
    FRAGMENTS_SAVED = "fragments_saved"
    ERROR = "error"


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
    qualifier: str | None = None
    note: str | None = None


class Fragment(BaseModel):
    fragment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # a unique uuid
    session_id: str  # (which session this came from)
    content: str  # your summary for searchable embedding
    exchange_ids: list[str]  # a list of exchange_id from the Exchange records that comprised this summary
    tags: list[Tag]  # tags
    timestamp: datetime


class Exchange(BaseModel):
    exchange_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    session_id: str | None = None  # back-reference to raw session
    human: Turn | None = None
    ai: Turn | None = None


class ClassifiedExchange(BaseModel):
    classification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str  # copied from the Exchange record
    exchange_ids: list[str]  # a list of exchange_id from the Exchange records that comprised this summary
    human_summary: str  # transcribe the relevant parts of the human message or copy it if it does not need condensing
    ai_summary: str  # transcribe the relevant parts of the AI message or copy it if it does not need condensing
    tags: list[Tag]  # from TAXONOMY classify this record using the TAXONOMY provided
    timestamp: datetime  # copied from the Exchange record


class ClassifiedExchangeList(BaseModel):
    exchanges: list[ClassifiedExchange]


class ThreadSegment(BaseModel):
    thread_name: str  # free-form snake_case, 2-6 words
    exchange_ids: list[str]  # which exchanges belong
    tags: list[Tag]  # from TAXONOMY classify this record using the TAXONOMY provided


class ThreadSegmentList(BaseModel):
    threads: list[ThreadSegment]


class ThreadClassificationResponse(BaseModel):
    tags: list[Tag]


class ExchangeClassificationRequest(BaseModel):
    exchange_id: str
    timestamp: datetime  # copied from the Exchange record
    dialog: str


class ExpandedThreadSegment(BaseModel):
    thread_name: str  # free-form snake_case, 2-6 words
    exchange_ids: list[str]  # which exchanges belong
    exchanges: list[ExchangeClassificationRequest]  # which exchanges belong
    tags: list[Tag]  # from TAXONOMY classify this record using the TAXONOMY provided


class FragmentDraft(BaseModel):
    """Lean fragment output from the extractor LLM — only reasoning decisions.

    Bookkeeping fields (session_id, fragment_id, timestamp) are filled in
    by the node code post-hoc, not by the LLM.
    """
    content: str  # the standalone, voice-preserving idea statement
    exchange_ids: list[str]  # which thread exchange(s) this idea came from
    tags: list[Tag]  # subset of the thread's tags that apply to this idea


class FragmentDraftList(BaseModel):
    fragments: list[FragmentDraft]


class Domain(BaseModel):
    tag: str  # from fixed taxonomy list
    score: float  # 0.0–1.0


class ScoreCard(BaseModel):
    question_score: float = Field(default=0, ge=0, le=1,
                                  description="Is the user asking a question?")  # 0.0–1.0  how much is this a request for information/opinion
    first_person_score: float = Field(default=0, ge=0, le=1,
                                      description="Is the user referring to himself?")  # 0.0–1.0  how much is the speaker talking about themselves
    personalization_score: float = Field(default=0, ge=0, le=1,
                                         description="Is the user asking for a change in the way the AI communicates? 0.0 = no such request, 1.0 = clear directive like 'call me Chris'")  # 0.0–1.0  is the user asking for a change in the ai behavior
    task_score: float = Field(default=0, ge=0, le=1,
                              description="Is the asking the AI to perform a specific task e.g. (please do this)")  # 0.0–1.0  how much does this contain an explicit directive
    domains: list[Domain] = Field(default_factory=list,
                                  description="Engagement score (0.0-1.0) for each taxonomy domain")  # scores across all 8 domains


class PromptKey(Enum):
    INTENT_CLASSIFIER = "intent_classifier"
    PROFILE_CLASSIFIER = "profile_classifier"
    PROFILE_SCANNER = "profile_classifier"
    CONVERSATION = "conversation"
    SOCRATIC = "socratic"
    GUIDANCE = "guidance"
    DECOMPOSER = "decomposer"
    THREAD_CLASSIFIER = "thread_classifier"
    EXCHANGE_CLASSIFIER = "exchange_classifier"
    FRAGMENT_EXTRACTOR = "extractor"


class ContextSpecification(BaseModel):
    prompt_key: PromptKey = Field(default=PromptKey.CONVERSATION)
    tags: list[str] = Field(default_factory=list)
    last_k_session_messages: int = Field(default=DEFAULT_RECENT_MESSAGES_COUNT, ge=0, le=20)
    last_k_recent_messages: int = Field(default=DEFAULT_SESSION_MESSAGES_COUNT, ge=0, le=20)
    top_k_retrieved_history: int = Field(default=DEFAULT_RETRIEVED_HISTORY_COUNT, ge=0, le=10)
    distance_retrieved_history: int = Field(default=DEFAULT_RETRIEVED_HISTORY_DISTANCE, ge=0, le=10)
    prompt_vars: dict[str, Any] = Field(default_factory=dict)


class UserProfile(BaseModel):
    user_id: str = Field(default="default_user", description="Unique identifier for the user - defaults to 'default_user'")
    # Communication
    response_style: str | None = Field(default=DEFAULT_RESPONSE_STYLE,
                                       description="Free text field describing how the AI should format and present data to the user")  # free-text, LLM-generated summary
    explanation_depth: str | None = Field(default=DEFAULT_EXPLANATION_DEPTH,
                                          description="Free text field describing the level that the AI should discuss at e.g. (expert, intermediate, advanced)")  # "expert" | "intermediate" | "beginner"
    tone: str | None = Field(default=DEFAULT_TONE,
                             description="Free text field describing the communication style the user prefers  .e.g. (friendly, formal, casual)")  # free-text

    # Domain
    interests: list[str] = Field(default_factory=lambda: list(DEFAULT_INTERESTS),
                                 description="Free text list of interests from specific user requests e.g. (please remember ia am interested in this subject <subject name>")  # accumulated across sessions
    active_projects: list[str] = Field(default_factory=list, description="Future - do not use")
    recurring_themes: list[str] = Field(default_factory=list, description="Future - do not use")

    # Interaction
    decision_style: str | None = Field(default=None,
                                       description="Free text field describing how the AI should reason about it's decisions e.g. (look for alternative, then propose one))")  # free-text
    learning_style: str | None = Field(default=DEFAULT_LEARNING_STYLE,
                                       description="Free text field describing how the AI should present information to the user e.g. (include alteratives and explain the reasoning)")  # free-text
    pet_peeves: list[str] = Field(default_factory=lambda: list(DEFAULT_PET_PEEVES),
                                  description="Free text list describing things the user expressly does NOT want")

    # Identity
    human_label: str = Field(default=HUMAN_NAME,
                             description="Free text field describing how the user wants to be addressed e.g. (please call me `Lord Vader`)")
    ai_label: str | None = Field(default=AI_NAME,
                                 description="Free text field describing how the user wants to address the AI e.g. (I'd like to refer to you as `Marvin`)")

    # Meta
    updated_at: datetime = Field(default_factory=datetime.now)
    is_updated: bool = Field(default=False,
                             description="Boolean flag describing whether the AI is submitting changes for publication to the user's profile or not")  # does the profile need to be saved
    is_current: bool = Field(default=False,
                             description="Boolean flag describing whether the user profile field needs to be re-evaluated and potentially changed")  # is the current version current - set if the user has indicated they want a change


"""
Insights
"""

class VerifierStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    FAILED = "failed"


class Insight(BaseModel):
    insight_id: str
    fragment_ids: list[str]  # provenance: REQUIRED
    label: str  # LLM-generated theme
    body: str  # the actual observation
    confidence: float
    created_at: datetime
    verifier_status: VerifierStatus


"""
Events
"""

class ExchangeEvent(BaseModel):
    topic: Literal["exchange"] = "exchange"
    payload: Exchange


class ClassifiedExchangeEvent(BaseModel):
    topic: Literal["exchange"] = "exchange"
    payload: Exchange


class FragmentEvent(BaseModel):
    topic: Literal["fragment"] = "fragment"
    payload: Fragment


JournalEvent = Annotated[
    Union[ExchangeEvent, ClassifiedExchangeEvent, FragmentEvent],
    Field(discriminator="topic"),
]
