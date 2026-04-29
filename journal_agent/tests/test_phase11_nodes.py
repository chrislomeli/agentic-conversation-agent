"""test_phase11_nodes.py — Behavioral unit tests for Phase 11 graph node factories.

Tests run against:
    - a FakeLLM that returns scripted structured responses without network calls
    - a FakeSubjectsRepo that tracks calls and returns scripted data

No Postgres or embedder is required. These tests exercise the node logic
(candidate filtering, vote construction, dedup, bookkeeping) independently
of the repo and LLM implementations.

Design doc: design/phase11-claim-based-insights.md
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest

from journal_agent.configure.config_builder import (
    COLD_START_SUBJECT_THRESHOLD,
    MIN_VOTE_STRENGTH,
    PROPOSER_DEDUP_SIMILARITY,
    SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH,
)
from journal_agent.graph.nodes.insight_nodes import (
    make_classify_stance,
    make_cluster_seed_subjects,
    make_persist_votes,
    make_propose_subject,
    make_route_candidates,
)
from journal_agent.graph.reflection_graph import should_cold_start
from journal_agent.graph.state import ReflectionState
from journal_agent.model.insights import (
    BatchStanceItem,
    BatchStanceResponse,
    CandidateSubject,
    Claim,
    FragmentProcessing,
    FragmentWorkItem,
    InitialVote,
    ProposedSubject,
    ProposerResponse,
    Stance,
    StanceVote,
    Subject,
    Vote,
)
from journal_agent.model.session import Cluster, ClusterList, Fragment, PromptKey


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fragment(
    content: str = "I meditate every morning without fail.",
    fragment_id: str = "frag-001",
    ts: datetime | None = None,
    embedding: list[float] | None = None,
    no_embedding: bool = False,
) -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        session_id="sess-test",
        content=content,
        exchange_ids=[],
        tags=[],
        embedding=[] if no_embedding else (embedding if embedding is not None else [0.1] * 384),
        timestamp=ts or datetime(2026, 1, 15, 9, 0),
    )


def _subject(label: str = "stance on meditation", subject_id: str = "sub-001") -> Subject:
    return Subject(subject_id=subject_id, label=label)


def _claim(subject_id: str = "sub-001", claim_id: str = "clm-001") -> Claim:
    return Claim(
        claim_id=claim_id,
        subject_id=subject_id,
        text="User is deeply committed to daily meditation.",
        version=1,
        is_current=True,
        embedding=[0.1] * 384,
    )


def _candidate(subject_id: str = "sub-001", similarity: float = 0.8) -> CandidateSubject:
    subj = _subject(subject_id=subject_id)
    clm = _claim(subject_id=subject_id)
    return CandidateSubject(subject=subj, current_claim=clm, similarity=similarity)


def _proposed_subject(label: str = "stance on solitude") -> ProposedSubject:
    return ProposedSubject(
        label=label,
        description="Tracks the user's relationship with alone time.",
        initial_claim="User actively seeks and values solitude.",
        initial_vote=InitialVote(stance=Stance.SUPPORT, strength=0.7, reasoning="Stated directly."),
    )


def _reflection_state(fragments: list[Fragment], work_items: list[FragmentWorkItem] | None = None) -> ReflectionState:
    return ReflectionState(
        session_id="test-session",
        fragments=fragments,
        work_items=work_items or [],
    )


# ── FakeLLM ───────────────────────────────────────────────────────────────────


class FakeLLM:
    """LLMClient stand-in.

    `astructured` is called ONCE per node invocation (outside the per-item loop).
    The returned runnable's `ainvoke` is called once per item. Responses are
    popped at `ainvoke` time so `astructured()` itself never exhausts the queue.
    """

    def __init__(self, responses: list[Any], model: str = "fake-model"):
        self._responses = list(responses)
        self._idx = 0
        self.model = model

    def astructured(self, schema: type):
        runnable = MagicMock()

        async def _ainvoke(messages):
            if self._idx >= len(self._responses):
                pytest.fail(f"FakeLLM exhausted — unexpected ainvoke call with {schema.__name__}")
            resp = self._responses[self._idx]
            self._idx += 1
            return resp

        runnable.ainvoke = _ainvoke
        return runnable


# ── FakeSubjectsRepo ──────────────────────────────────────────────────────────


class FakeSubjectsRepo:
    """In-memory SubjectsRepository stand-in for node tests."""

    def __init__(
        self,
        candidate_results: list[tuple[Subject, Claim, float]] | None = None,
        active_with_claims: list[tuple[Subject, Claim]] | None = None,
        created_subject: tuple[Subject, Claim, Vote] | None = None,
        active_count: int = 0,
    ):
        self._candidates = candidate_results or []
        self._active_with_claims = active_with_claims or []
        self._created = created_subject
        self._active_count = active_count
        self.inserted_votes: list[list[Vote]] = []
        self.processing_records: list[dict] = []
        self.create_calls: list[tuple] = []

    def count_active_subjects(self) -> int:
        return self._active_count

    def search_candidate_subjects(self, embedding, top_k, min_similarity):
        # Must return list[tuple[Subject, Claim, float]] — same shape as the real repo.
        return self._candidates

    def list_active_subjects_with_claims(self, limit: int = 100):
        return self._active_with_claims

    def create_subject_with_claim(self, proposed: ProposedSubject, first_vote: Vote):
        self.create_calls.append((proposed, first_vote))
        if self._created:
            return self._created
        subj = Subject(label=proposed.label)
        clm = Claim(
            subject_id=subj.subject_id,
            text=proposed.initial_claim,
            version=1,
            is_current=True,
        )
        vote = first_vote.model_copy(update={"subject_id": subj.subject_id, "claim_id": clm.claim_id})
        return subj, clm, vote

    def insert_votes(self, votes: list[Vote]) -> None:
        self.inserted_votes.append(list(votes))

    def record_processing(self, fragment_id, model_signature, vote_count, status=None, error_detail=None):
        self.processing_records.append(
            {"fragment_id": fragment_id, "vote_count": vote_count, "model_signature": model_signature}
        )
        return FragmentProcessing(fragment_id=fragment_id, model_signature=model_signature, vote_count=vote_count)

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(384, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# make_route_candidates
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_route_candidates_produces_one_item_per_fragment():
    repo = FakeSubjectsRepo(candidate_results=[])
    node = make_route_candidates(repo)
    fragments = [_fragment(fragment_id=f"frag-{i:03d}") for i in range(3)]
    state = _reflection_state(fragments)

    result = await node(state)

    assert len(result["work_items"]) == 3
    for item, frag in zip(result["work_items"], fragments):
        assert item.fragment.fragment_id == frag.fragment_id


@pytest.mark.asyncio
async def test_route_candidates_populates_candidates_from_search():
    candidate = _candidate()
    repo = FakeSubjectsRepo(candidate_results=[
        (candidate.subject, candidate.current_claim, candidate.similarity)
    ])
    node = make_route_candidates(repo)
    state = _reflection_state([_fragment()])

    result = await node(state)

    items = result["work_items"]
    assert len(items[0].candidates) == 1
    assert items[0].candidates[0].subject.subject_id == "sub-001"


@pytest.mark.asyncio
async def test_route_candidates_skips_search_for_fragment_without_embedding():
    c = _candidate()
    repo = FakeSubjectsRepo(candidate_results=[(c.subject, c.current_claim, c.similarity)])
    node = make_route_candidates(repo)
    state = _reflection_state([_fragment(no_embedding=True)])

    result = await node(state)

    assert result["work_items"][0].candidates == []


@pytest.mark.asyncio
async def test_route_candidates_empty_fragments_returns_empty():
    repo = FakeSubjectsRepo()
    node = make_route_candidates(repo)
    state = _reflection_state([])

    result = await node(state)

    assert result["work_items"] == []


# ═════════════════════════════════════════════════════════════════════════════
# make_classify_stance
# ═════════════════════════════════════════════════════════════════════════════


def _item_with_candidate(fragment_id: str = "frag-001") -> FragmentWorkItem:
    return FragmentWorkItem(fragment=_fragment(fragment_id=fragment_id), candidates=[_candidate()])


@pytest.mark.asyncio
async def test_classify_stance_maps_response_to_votes():
    response = BatchStanceResponse(results=[
        BatchStanceItem(fragment_id="frag-001", votes=[
            StanceVote(subject_id="sub-001", stance=Stance.SUPPORT, strength=0.8, reasoning="daily practice"),
        ]),
    ])
    llm = FakeLLM([response])
    node = make_classify_stance(llm)
    state = _reflection_state([], work_items=[_item_with_candidate()])

    result = await node(state)

    items = result["work_items"]
    assert len(items[0].votes) == 1
    vote = items[0].votes[0]
    assert vote.stance == Stance.SUPPORT
    assert vote.strength == 0.8
    assert vote.subject_id == "sub-001"
    assert vote.claim_id == "clm-001"
    assert vote.fragment_id == "frag-001"


@pytest.mark.asyncio
async def test_classify_stance_filters_votes_below_min_strength():
    below = MIN_VOTE_STRENGTH - 0.01
    response = BatchStanceResponse(results=[
        BatchStanceItem(fragment_id="frag-001", votes=[
            StanceVote(subject_id="sub-001", stance=Stance.SUPPORT, strength=below, reasoning="weak"),
        ]),
    ])
    llm = FakeLLM([response])
    node = make_classify_stance(llm)
    state = _reflection_state([], work_items=[_item_with_candidate()])

    result = await node(state)

    assert result["work_items"][0].votes == []


@pytest.mark.asyncio
async def test_classify_stance_skips_llm_when_no_candidates():
    llm = FakeLLM([])  # empty — any call would raise StopIteration
    node = make_classify_stance(llm)
    item = FragmentWorkItem(fragment=_fragment(), candidates=[])
    state = _reflection_state([], work_items=[item])

    result = await node(state)

    assert result["work_items"][0].votes == []


@pytest.mark.asyncio
async def test_classify_stance_allows_ambivalent_votes():
    response = BatchStanceResponse(results=[
        BatchStanceItem(fragment_id="frag-001", votes=[
            StanceVote(subject_id="sub-001", stance=Stance.SUPPORT, strength=0.7, reasoning="one thing"),
            StanceVote(subject_id="sub-001", stance=Stance.CONTRADICT, strength=0.6, reasoning="another"),
        ]),
    ])
    llm = FakeLLM([response])
    node = make_classify_stance(llm)
    state = _reflection_state([], work_items=[_item_with_candidate()])

    result = await node(state)

    assert len(result["work_items"][0].votes) == 2


@pytest.mark.asyncio
async def test_classify_stance_returns_item_unchanged_on_llm_error():
    llm = MagicMock()
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    llm.astructured = MagicMock(return_value=runnable)
    llm.model = "fake"
    node = make_classify_stance(llm)
    state = _reflection_state([], work_items=[_item_with_candidate()])

    result = await node(state)

    assert result["work_items"][0].votes == []


# ═════════════════════════════════════════════════════════════════════════════
# make_propose_subject
# ═════════════════════════════════════════════════════════════════════════════


def _item_no_votes() -> FragmentWorkItem:
    return FragmentWorkItem(fragment=_fragment(), candidates=[])


def _item_strong_vote() -> FragmentWorkItem:
    vote = Vote(
        subject_id="sub-001",
        claim_id="clm-001",
        fragment_id="frag-001",
        stance=Stance.SUPPORT,
        strength=SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH + 0.1,
        reasoning="explicit",
        fragment_dated_at=datetime(2026, 1, 15),
        model_signature="test/v1",
    )
    return FragmentWorkItem(fragment=_fragment(), candidates=[], votes=[vote])


def _item_weak_vote() -> FragmentWorkItem:
    vote = Vote(
        subject_id="sub-001",
        claim_id="clm-001",
        fragment_id="frag-001",
        stance=Stance.SUPPORT,
        strength=SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH - 0.1,
        reasoning="weak signal",
        fragment_dated_at=datetime(2026, 1, 15),
        model_signature="test/v1",
    )
    return FragmentWorkItem(fragment=_fragment(), candidates=[], votes=[vote])


@pytest.mark.asyncio
async def test_propose_subject_sets_proposed_when_llm_returns_subject():
    proposed = _proposed_subject()
    repo = FakeSubjectsRepo()
    llm = FakeLLM([ProposerResponse(new_subject=proposed)])
    node = make_propose_subject(llm, repo)
    state = _reflection_state([], work_items=[_item_no_votes()])

    result = await node(state)

    item = result["work_items"][0]
    assert item.proposed_subject is not None
    assert item.proposed_subject.label == "stance on solitude"


@pytest.mark.asyncio
async def test_propose_subject_skips_when_llm_returns_null():
    repo = FakeSubjectsRepo()
    llm = FakeLLM([ProposerResponse(new_subject=None)])
    node = make_propose_subject(llm, repo)
    state = _reflection_state([], work_items=[_item_no_votes()])

    result = await node(state)

    assert result["work_items"][0].proposed_subject is None


@pytest.mark.asyncio
async def test_propose_subject_skips_item_with_strong_existing_vote():
    repo = FakeSubjectsRepo()
    llm = FakeLLM([])  # no calls should happen
    node = make_propose_subject(llm, repo)
    state = _reflection_state([], work_items=[_item_strong_vote()])

    result = await node(state)

    assert result == {}


@pytest.mark.asyncio
async def test_propose_subject_triggers_for_weak_votes():
    proposed = _proposed_subject()
    repo = FakeSubjectsRepo()
    llm = FakeLLM([ProposerResponse(new_subject=proposed)])
    node = make_propose_subject(llm, repo)
    state = _reflection_state([], work_items=[_item_weak_vote()])

    result = await node(state)

    assert result["work_items"][0].proposed_subject is not None


# ═════════════════════════════════════════════════════════════════════════════
# make_persist_votes
# ═════════════════════════════════════════════════════════════════════════════


def _item_with_vote(fragment_id: str = "frag-001", subject_id: str = "sub-001") -> FragmentWorkItem:
    vote = Vote(
        subject_id=subject_id,
        claim_id="clm-001",
        fragment_id=fragment_id,
        stance=Stance.SUPPORT,
        strength=0.8,
        reasoning="...",
        fragment_dated_at=datetime(2026, 1, 15),
        model_signature="fake-model/stance_classifier/v1",
    )
    return FragmentWorkItem(fragment=_fragment(fragment_id=fragment_id), candidates=[], votes=[vote])


@pytest.mark.asyncio
async def test_persist_votes_inserts_votes_and_records_processing():
    repo = FakeSubjectsRepo()
    llm = MagicMock()
    llm.model = "fake-model"
    node = make_persist_votes(repo, llm)
    item = _item_with_vote()
    state = _reflection_state([], work_items=[item])

    await node(state)

    assert len(repo.inserted_votes) == 1
    assert len(repo.inserted_votes[0]) == 1
    assert len(repo.processing_records) == 1
    assert repo.processing_records[0]["vote_count"] == 1
    assert repo.processing_records[0]["fragment_id"] == "frag-001"


@pytest.mark.asyncio
async def test_persist_votes_creates_subject_for_proposal():
    repo = FakeSubjectsRepo()
    llm = MagicMock()
    llm.model = "fake-model"
    node = make_persist_votes(repo, llm)
    item = FragmentWorkItem(
        fragment=_fragment(),
        proposed_subject=_proposed_subject(),
    )
    state = _reflection_state([], work_items=[item])

    await node(state)

    assert len(repo.create_calls) == 1
    assert repo.processing_records[0]["vote_count"] == 1


@pytest.mark.asyncio
async def test_persist_votes_deduplicates_identical_proposals():
    """Two work items proposing similar subjects: only one create_subject_with_claim,
    the follower's initial_vote goes into insert_votes, and BOTH items get
    processing records with vote_count=1."""
    # Repo returns normalized vectors; dot product = 1.0 > PROPOSER_DEDUP_SIMILARITY.
    repo = FakeSubjectsRepo()
    llm = MagicMock()
    llm.model = "fake-model"
    node = make_persist_votes(repo, llm)

    item_a = FragmentWorkItem(
        fragment=_fragment(fragment_id="frag-001"),
        proposed_subject=_proposed_subject("stance on solitude"),
    )
    item_b = FragmentWorkItem(
        fragment=_fragment(fragment_id="frag-002"),
        proposed_subject=_proposed_subject("stance on solitude"),
    )
    state = _reflection_state([], work_items=[item_a, item_b])

    await node(state)

    # Only one subject created.
    assert len(repo.create_calls) == 1

    # Follower's vote appears in insert_votes.
    all_inserted = [v for batch in repo.inserted_votes for v in batch]
    assert len(all_inserted) == 1  # the follower redirect

    # Both fragments get a processing record with vote_count=1.
    counts = {r["fragment_id"]: r["vote_count"] for r in repo.processing_records}
    assert counts["frag-001"] == 1  # canonical
    assert counts["frag-002"] == 1  # follower — this was the bug


@pytest.mark.asyncio
async def test_persist_votes_noop_on_empty_items():
    repo = FakeSubjectsRepo()
    llm = MagicMock()
    llm.model = "fake-model"
    node = make_persist_votes(repo, llm)
    state = _reflection_state([], work_items=[])

    await node(state)

    assert repo.inserted_votes == []
    assert repo.processing_records == []


@pytest.mark.asyncio
async def test_persist_votes_distinct_proposals_both_created():
    """Two proposals that are NOT similar should both be created."""
    # embed_text returns ones — all proposals will appear similar with default impl.
    # Override to return distinct embeddings per call.
    call_count = 0

    def distinct_embed(text: str) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        vec = np.zeros(384, dtype=np.float32)
        vec[call_count - 1] = 1.0  # orthogonal vectors
        return vec

    repo = FakeSubjectsRepo()
    repo.embed_text = distinct_embed
    llm = MagicMock()
    llm.model = "fake-model"
    node = make_persist_votes(repo, llm)

    item_a = FragmentWorkItem(
        fragment=_fragment(fragment_id="frag-001"),
        proposed_subject=_proposed_subject("stance on solitude"),
    )
    item_b = FragmentWorkItem(
        fragment=_fragment(fragment_id="frag-002"),
        proposed_subject=_proposed_subject("relationship with father"),
    )
    state = _reflection_state([], work_items=[item_a, item_b])

    await node(state)

    assert len(repo.create_calls) == 2


# ═════════════════════════════════════════════════════════════════════════════
# make_cluster_seed_subjects (cold-start path)
# ═════════════════════════════════════════════════════════════════════════════


def _cluster(fragment_ids: list[str], label: str = "test cluster") -> Cluster:
    return Cluster(fragment_ids=fragment_ids, label=label, score=0.8)


@pytest.mark.asyncio
async def test_cluster_seed_subjects_empty_fragments_completes():
    repo = FakeSubjectsRepo()
    llm = FakeLLM([])
    node = make_cluster_seed_subjects(llm, repo)
    state = _reflection_state([])

    result = await node(state)

    from journal_agent.model.session import StatusValue as SV
    assert result["status"] == SV.COMPLETED
    assert repo.create_calls == []
    assert repo.processing_records == []


@pytest.mark.asyncio
async def test_cluster_seed_subjects_creates_subject_per_cluster():
    """Two clusters → two subjects, every fragment recorded as processed."""
    fragments = [
        _fragment(fragment_id="frag-001"),
        _fragment(fragment_id="frag-002"),
        _fragment(fragment_id="frag-003"),
    ]
    cluster_response = ClusterList(clusters=[
        _cluster(["frag-001", "frag-002"], label="cluster A"),
        _cluster(["frag-003"], label="cluster B"),
    ])
    seed_a = ProposerResponse(new_subject=_proposed_subject("stance on A"))
    seed_b = ProposerResponse(new_subject=_proposed_subject("stance on B"))

    llm = FakeLLM([cluster_response, seed_a, seed_b])
    repo = FakeSubjectsRepo()
    node = make_cluster_seed_subjects(llm, repo)
    state = _reflection_state(fragments)

    await node(state)

    assert len(repo.create_calls) == 2
    # frag-002 gets a follow-up vote (it's not the anchor); frag-003 is solo so no follow-ups
    inserted = [v for batch in repo.inserted_votes for v in batch]
    assert len(inserted) == 1
    assert inserted[0].fragment_id == "frag-002"
    assert {r["fragment_id"] for r in repo.processing_records} == {"frag-001", "frag-002", "frag-003"}
    counts = {r["fragment_id"]: r["vote_count"] for r in repo.processing_records}
    assert counts["frag-001"] == 1
    assert counts["frag-002"] == 1
    assert counts["frag-003"] == 1


@pytest.mark.asyncio
async def test_cluster_seed_subjects_skips_null_proposals_but_records_processing():
    """LLM rejects a cluster (null) — fragments still recorded as processed (vote_count=0)."""
    fragments = [_fragment(fragment_id="frag-001"), _fragment(fragment_id="frag-002")]
    cluster_response = ClusterList(clusters=[_cluster(["frag-001", "frag-002"])])
    seed_null = ProposerResponse(new_subject=None)

    llm = FakeLLM([cluster_response, seed_null])
    repo = FakeSubjectsRepo()
    node = make_cluster_seed_subjects(llm, repo)
    state = _reflection_state(fragments)

    await node(state)

    assert repo.create_calls == []
    assert repo.inserted_votes == []
    counts = {r["fragment_id"]: r["vote_count"] for r in repo.processing_records}
    assert counts["frag-001"] == 0
    assert counts["frag-002"] == 0


@pytest.mark.asyncio
async def test_cluster_seed_subjects_marks_outliers_processed():
    """Fragments not assigned to any cluster are marked processed with vote_count=0."""
    fragments = [
        _fragment(fragment_id="frag-001"),
        _fragment(fragment_id="frag-002"),
        _fragment(fragment_id="frag-003"),
    ]
    cluster_response = ClusterList(clusters=[_cluster(["frag-001", "frag-002"])])
    seed = ProposerResponse(new_subject=_proposed_subject())

    llm = FakeLLM([cluster_response, seed])
    repo = FakeSubjectsRepo()
    node = make_cluster_seed_subjects(llm, repo)
    state = _reflection_state(fragments)

    await node(state)

    counts = {r["fragment_id"]: r["vote_count"] for r in repo.processing_records}
    assert counts["frag-003"] == 0  # outlier
    assert counts["frag-001"] == 1  # cluster member
    assert counts["frag-002"] == 1  # cluster member


@pytest.mark.asyncio
async def test_cluster_seed_subjects_clustering_failure_returns_error():
    """If the clustering LLM call fails, the node returns ERROR without DB writes."""
    fragments = [_fragment()]
    llm = MagicMock()
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    llm.astructured = MagicMock(return_value=runnable)
    llm.model = "fake-model"

    repo = FakeSubjectsRepo()
    node = make_cluster_seed_subjects(llm, repo)
    state = _reflection_state(fragments)

    result = await node(state)

    from journal_agent.model.session import StatusValue as SV
    assert result["status"] == SV.ERROR
    assert repo.create_calls == []
    assert repo.processing_records == []


# ═════════════════════════════════════════════════════════════════════════════
# should_cold_start router
# ═════════════════════════════════════════════════════════════════════════════


def test_should_cold_start_routes_to_cluster_when_subjects_sparse():
    repo = FakeSubjectsRepo(active_count=COLD_START_SUBJECT_THRESHOLD - 1)
    router = should_cold_start(repo)
    assert router(_reflection_state([])) == "cluster_seed_subjects"


def test_should_cold_start_routes_to_route_candidates_when_at_threshold():
    repo = FakeSubjectsRepo(active_count=COLD_START_SUBJECT_THRESHOLD)
    router = should_cold_start(repo)
    assert router(_reflection_state([])) == "route_candidates"


def test_should_cold_start_falls_back_to_route_candidates_on_count_failure():
    repo = FakeSubjectsRepo()
    repo.count_active_subjects = MagicMock(side_effect=RuntimeError("DB down"))
    router = should_cold_start(repo)
    assert router(_reflection_state([])) == "route_candidates"
