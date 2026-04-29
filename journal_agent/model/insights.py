"""insights.py — Phase 11 domain models for claim-based insight tracking.

Three persistent entities (Subject, Claim, Vote) plus a bookkeeping entity
(FragmentProcessing). Their mutability profiles differ:

    Subject              — identity stable; label/status/last_activity mutable
    Claim                — append a new version; flip is_current
    Vote                 — append-only; can be invalidated, never edited
    FragmentProcessing   — append-only

Plus the LLM-facing structured-output models used by the three new prompts:

    StanceResponse       — stance_classifier output
    ProposerResponse     — subject_proposer output
    RegeneratorResponse  — claim_regenerator output

And the in-flight work-item shapes that flow through the reflection graph:

    CandidateSubject     — Subject + its current Claim + similarity score
    FragmentWorkItem     — one fragment's accumulating per-node state

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from journal_agent.model.session import Fragment

# ── Persistent entities ──────────────────────────────────────────────────


class SubjectStatus(StrEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    SUPERSEDED = "superseded"
    MERGED = "merged"


class Stance(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"


class ProcessingStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class Subject(BaseModel):
    """A stable handle for a tracked idea. Identity persists; label may evolve.

    Forks set parent_subject_id; soft merges set merged_into_id (queries union).
    """

    subject_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(description="Short, human-readable, e.g. 'stance on Buddhism'.")
    description: str | None = Field(
        default=None, description="Optional longer gloss describing what this subject tracks."
    )
    status: SubjectStatus = Field(default=SubjectStatus.ACTIVE)
    parent_subject_id: str | None = Field(
        default=None,
        description="Set when this subject was forked from another (refinement/split).",
    )
    merged_into_id: str | None = Field(
        default=None,
        description="Set when this subject was soft-merged into another. Queries should union both.",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity_at: datetime = Field(
        default_factory=datetime.now,
        description="Denormalized; updated on each new vote against this subject.",
    )


class Claim(BaseModel):
    """The LLM's current best phrasing of the user's position on a subject.

    Versioned; one row per subject has is_current=True. The embedding lives
    here (not on Subject) because the claim text is the semantically rich part
    used for routing new fragments to candidate subjects.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str
    text: str = Field(description="Current best phrasing of the user's position.")
    version: int = Field(ge=1, description="Monotonic per subject; first claim is version 1.")
    is_current: bool = Field(default=False)
    embedding: list[float] = Field(
        default_factory=list,
        description="Vector embedding of `label + text`. NULL until embedded at save time.",
    )
    regenerated_at_vote_count: int = Field(
        default=0,
        description="Total votes against the subject when this claim version was generated. "
                    "Drives the 'regenerate when N new votes have accumulated' trigger.",
    )
    created_at: datetime = Field(default_factory=datetime.now)


class Vote(BaseModel):
    """Append-only timestamped evidence that a fragment supports/contradicts a subject.

    Attaches to subject_id (stable) for efficient traction queries. claim_id records
    which claim version was evaluated against — useful when the claim drifts.
    fragment_dated_at is the user-write timestamp; all belief queries filter on
    this. processed_at is for audit only.
    """

    vote_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str
    claim_id: str = Field(description="The claim that was evaluated against when this vote was cast.")
    fragment_id: str
    stance: Stance
    strength: float = Field(ge=0.0, le=1.0, description="LLM confidence in the vote (>=0.3 floor).")
    reasoning: str = Field(description="LLM explanation citing the fragment passage.")
    fragment_dated_at: datetime = Field(
        description="When the user wrote the fragment. Drives all 'as-of' belief queries."
    )
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="When the vote was cast. Audit only; not used for belief queries.",
    )
    model_signature: str = Field(
        description="Identifies model + prompt version; pairs with FragmentProcessing.model_signature."
    )
    signals: dict | None = Field(
        default=None,
        description="Extra context (length, time-of-day, prompt context) for future weighting strategies.",
    )
    invalidated_at: datetime | None = Field(default=None)
    invalidation_reason: str | None = Field(default=None)


class FragmentProcessing(BaseModel):
    """Bookkeeping for the reflect node's per-fragment processing loop.

    A row exists per (fragment, model_signature, processing run). Zero votes is
    a valid successful outcome — distinguishable from 'haven't looked yet'.
    """

    processing_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fragment_id: str
    processed_at: datetime = Field(default_factory=datetime.now)
    model_signature: str
    vote_count: int = Field(default=0, ge=0)
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS)
    error_detail: str | None = Field(default=None)


# ── Computed summaries (not persisted) ───────────────────────────────────


class SubjectSnapshot(BaseModel):
    """Computed summary of a subject's current claim and vote traction.

    Built by the claim_reflect_node after running the Phase 11 pipeline.
    Combines Subject.label + Claim.text + aggregated vote counts into a
    single serialisable record for injection into the AI context prompt.
    Not persisted — reconstructed fresh each time /reflect2 is called.
    """

    label: str
    claim: str
    traction: float
    support: int
    contradict: int


# ── LLM-facing structured outputs ────────────────────────────────────────


class StanceVote(BaseModel):
    """One vote produced by the stance classifier for a single candidate subject."""

    subject_id: str = Field(description="ID of the candidate subject this vote is cast against.")
    stance: Stance
    strength: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0.9 = explicit and unambiguous; 0.6 = clearly implied; 0.3 = weak inference. "
            "Do not return strengths below 0.3 — omit the vote instead."
        ),
    )
    reasoning: str = Field(
        description="Quote or paraphrase the specific fragment passage that drives the vote."
    )


class StanceResponse(BaseModel):
    """Output of the stance_classifier prompt. Empty list is valid and common."""

    votes: list[StanceVote] = Field(default_factory=list)


class InitialVote(BaseModel):
    """The first vote bundled with a newly proposed subject."""

    stance: Stance
    strength: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ProposedSubject(BaseModel):
    """A new subject proposed by the LLM, bundled with its first vote."""

    label: str = Field(description="Short stable name, e.g. 'stance on solitude'.")
    description: str = Field(description="One sentence describing what this tracks.")
    initial_claim: str = Field(description="Current best phrasing of the user's position.")
    initial_vote: InitialVote


class ProposerResponse(BaseModel):
    """Output of the subject_proposer prompt. None means 'do not create'."""

    new_subject: ProposedSubject | None = None


class BatchStanceItem(BaseModel):
    """Per-fragment result returned by the batch stance classifier."""

    fragment_id: str
    votes: list[StanceVote] = Field(default_factory=list)


class BatchStanceResponse(BaseModel):
    """Output of the stance_classifier_batch prompt. One result per input item."""

    results: list[BatchStanceItem] = Field(default_factory=list)


class BatchVerifierItem(BaseModel):
    """Per-insight result returned by the batch verifier."""

    insight_id: str
    verifier_score: float = Field(ge=0.0, le=1.0)
    verifier_comments: str


class BatchVerifierResponse(BaseModel):
    """Output of the verify_insights_batch prompt. One result per input item."""

    results: list[BatchVerifierItem] = Field(default_factory=list)


class RegeneratorAction(StrEnum):
    NO_CHANGE = "no_change"
    REWRITE = "rewrite"
    FORK_SUGGESTED = "fork_suggested"


class RegeneratorResponse(BaseModel):
    """Output of the claim_regenerator prompt."""

    action: RegeneratorAction
    new_claim_text: str | None = Field(
        default=None,
        description="New phrasing reflecting the user's CURRENT state. Required when action='rewrite'.",
    )
    change_summary: str | None = Field(
        default=None,
        description="What evolved and why. Required when action='rewrite'.",
    )
    fork_reasoning: str | None = Field(
        default=None,
        description="Explanation of why the subject should fork. Required when action='fork_suggested'.",
    )


# ── Reflection-graph work items ──────────────────────────────────────────


class CandidateSubject(BaseModel):
    """A candidate subject for a fragment, surfaced by the routing step.

    Bundles the Subject (identity), its current Claim (the text the LLM will
    actually vote against), and the cosine similarity score from the routing
    search. Carrying all three forward avoids a re-fetch in classify_stance
    and keeps the routing's relevance signal available for downstream weighting.
    """

    subject: Subject
    current_claim: Claim
    similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the fragment's embedding and the current claim's embedding.",
    )


class FragmentWorkItem(BaseModel):
    """One fragment's accumulating state as it flows through the reflection graph.

    Each node maps over `state.work_items` in parallel and enriches one
    dimension of each item:

        route_candidates    → fills `candidates`
        classify_stance     → fills `votes` (only against EXISTING subjects)
        propose_subject     → fills `proposed_subject` (when the LLM proposes a new one)
        persist_votes       → reads everything; materializes proposed_subject's
                              initial_vote into a real Vote at write time

    Note: votes for newly-proposed subjects do NOT live in `votes` — they ride
    inside `proposed_subject.initial_vote` until persist time, when the new
    subject finally has a real subject_id. This keeps `votes` cleanly typed
    (every Vote here references an existing subject row).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fragment: Fragment
    candidates: list[CandidateSubject] = Field(default_factory=list)
    votes: list[Vote] = Field(default_factory=list)
    proposed_subject: ProposedSubject | None = None
