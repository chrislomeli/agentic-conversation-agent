import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum

from pydantic import BaseModel, Field

from journal_agent.configure.config_builder import DEFAULT_RECENT_MESSAGES_COUNT, DEFAULT_SESSION_MESSAGES_COUNT, \
    DEFAULT_RETRIEVED_HISTORY_COUNT, DEFAULT_RETRIEVED_HISTORY_DISTANCE


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
    session_id: str  # (which Turn this came from)
    content: str  # your summary for searchable embedding
    exchange_ids: list[str]  # a list of exchange_id from the Exchange records that comprised this summary
    tags: list[Tag]  # tags
    timestamp: datetime


class FragmentList(BaseModel):
    exchanges: list[Fragment]


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


class ExpandedThreadSegmentList(BaseModel):
    threads: list[ExpandedThreadSegment]


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
    question_score: float  # 0.0–1.0  how much is this a request for information/opinion
    first_person_score: float  # 0.0–1.0  how much is the speaker talking about themselves
    task_score: float  # 0.0–1.0  how much does this contain an explicit directive
    domains: list[Domain]  # scores across all 8 domains


class PromptKey(Enum):
    INTENT_CLASSIFIER = "intent_classifier"
    CONVERSATION = "conversation"
    SOCRATIC = "socratic"
    GUIDANCE = "guidance"
    DECOMPOSER = "decomposer"
    THREAD_CLASSIFIER = "thread_classifier"
    EXCHANGE_CLASSIFIER = "exchange_classifier"
    FRAGMENT_EXTRACTOR = "extractor"


class ContextSpecification(BaseModel):
    prompt_key: PromptKey = Field(default=PromptKey.CONVERSATION)
    tags: list[str] = Field(default=[])
    last_k_session_messages: int = Field(default=DEFAULT_RECENT_MESSAGES_COUNT, ge=0, le=20)
    last_k_recent_messages: int = Field(default=DEFAULT_SESSION_MESSAGES_COUNT, ge=0, le=20)
    top_k_retrieved_history: int = Field(default=DEFAULT_RETRIEVED_HISTORY_COUNT, ge=0, le=10)
    distance_retrieved_history: int = Field(default=DEFAULT_RETRIEVED_HISTORY_DISTANCE, ge=0, le=10)