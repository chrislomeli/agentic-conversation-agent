"""Skeleton smoke tests for Phase 11 — claim-based insights.

Verifies that the new graph compiles, runs end-to-end against an empty
ReflectionState without errors, and that the new pydantic models round-trip.
This file does NOT test behavior — it tests that the scaffolding is coherent.

Real behavioral tests land alongside each implementation PR.

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from journal_agent.configure.prompts import get_prompt, get_prompt_version
from journal_agent.graph.reflection_graph import build_claim_reflection_graph
from journal_agent.graph.state import ReflectionState
from journal_agent.model.insights import (
    Claim,
    FragmentProcessing,
    InitialVote,
    ProcessingStatus,
    ProposedSubject,
    ProposerResponse,
    RegeneratorAction,
    RegeneratorResponse,
    Stance,
    StanceResponse,
    StanceVote,
    Subject,
    SubjectStatus,
    Vote,
)
from journal_agent.model.session import PromptKey


# ── Pydantic round-trips ──────────────────────────────────────────────────


def test_subject_round_trips():
    s = Subject(label="stance on solitude")
    raw = s.model_dump(mode="json")
    s2 = Subject.model_validate(raw)
    assert s2.label == "stance on solitude"
    assert s2.status == SubjectStatus.ACTIVE


def test_claim_round_trips():
    c = Claim(subject_id="sub-1", text="user values quiet alone time", version=1, is_current=True)
    raw = c.model_dump(mode="json")
    c2 = Claim.model_validate(raw)
    assert c2.text == "user values quiet alone time"
    assert c2.version == 1
    assert c2.is_current is True


def test_vote_round_trips():
    from datetime import datetime
    v = Vote(
        subject_id="sub-1",
        fragment_id="frag-1",
        stance=Stance.SUPPORT,
        strength=0.7,
        reasoning="user explicitly stated...",
        claim_version_at_vote=1,
        fragment_dated_at=datetime(2026, 1, 15),
        model_signature="claude-sonnet-4-5/stance_classifier_v1",
    )
    raw = v.model_dump(mode="json")
    v2 = Vote.model_validate(raw)
    assert v2.stance == Stance.SUPPORT
    assert v2.strength == 0.7


def test_fragment_processing_round_trips():
    fp = FragmentProcessing(
        fragment_id="frag-1",
        model_signature="claude-sonnet-4-5/stance_classifier_v1",
        vote_count=2,
    )
    raw = fp.model_dump(mode="json")
    fp2 = FragmentProcessing.model_validate(raw)
    assert fp2.status == ProcessingStatus.SUCCESS
    assert fp2.vote_count == 2


def test_stance_response_default_is_empty():
    sr = StanceResponse()
    assert sr.votes == []


def test_stance_response_with_votes():
    sr = StanceResponse(votes=[
        StanceVote(subject_id="sub-1", stance=Stance.SUPPORT, strength=0.8, reasoning="..."),
    ])
    raw = sr.model_dump(mode="json")
    sr2 = StanceResponse.model_validate(raw)
    assert len(sr2.votes) == 1
    assert sr2.votes[0].stance == Stance.SUPPORT


def test_proposer_response_null():
    pr = ProposerResponse()
    assert pr.new_subject is None


def test_proposer_response_with_subject():
    pr = ProposerResponse(new_subject=ProposedSubject(
        label="relationship with father",
        description="...",
        initial_claim="...",
        initial_vote=InitialVote(stance=Stance.SUPPORT, strength=0.9, reasoning="..."),
    ))
    raw = pr.model_dump(mode="json")
    pr2 = ProposerResponse.model_validate(raw)
    assert pr2.new_subject.label == "relationship with father"


def test_regenerator_response_actions():
    no_change = RegeneratorResponse(action=RegeneratorAction.NO_CHANGE)
    rewrite = RegeneratorResponse(
        action=RegeneratorAction.REWRITE,
        new_claim_text="updated claim",
        change_summary="user has shifted...",
    )
    fork = RegeneratorResponse(
        action=RegeneratorAction.FORK_SUGGESTED,
        fork_reasoning="evidence has split...",
    )
    for r in (no_change, rewrite, fork):
        raw = r.model_dump(mode="json")
        RegeneratorResponse.model_validate(raw)


# ── Prompt registry ───────────────────────────────────────────────────────


def test_phase11_prompts_are_registered():
    for key in (
        PromptKey.STANCE_CLASSIFIER,
        PromptKey.SUBJECT_PROPOSER,
        PromptKey.CLAIM_REGENERATOR,
    ):
        text = get_prompt(key)
        assert isinstance(text, str)
        assert len(text) > 100  # non-empty, real content
        assert get_prompt_version(key) == "v1"


def test_stance_prompt_contains_load_bearing_instructions():
    text = get_prompt(PromptKey.STANCE_CLASSIFIER)
    assert "Silence is the default" in text or "SILENCE IS THE DEFAULT" in text
    assert "0.3" in text  # strength floor


def test_proposer_prompt_biases_against_creation():
    text = get_prompt(PromptKey.SUBJECT_PROPOSER)
    assert "BIAS TOWARD NOT CREATING" in text or "bias toward NOT creating" in text.lower()


def test_regenerator_prompt_calls_out_trajectory():
    text = get_prompt(PromptKey.CLAIM_REGENERATOR)
    assert "CURRENT" in text or "current" in text.lower()
    assert "fork" in text.lower()


# ── Graph compiles + runs end-to-end on empty state ──────────────────────


@pytest.fixture
def fake_registry():
    """A registry whose .get(...) returns a MagicMock LLM client. The skeleton
    nodes never invoke the LLM, so this is fine."""
    reg = MagicMock()
    reg.get.return_value = MagicMock()
    return reg


@pytest.fixture
def fake_subjects_repo():
    return MagicMock()


def test_claim_reflection_graph_compiles(fake_registry, fake_subjects_repo):
    graph = build_claim_reflection_graph(
        registry=fake_registry,
        subjects_repo=fake_subjects_repo,
    )
    assert graph is not None


@pytest.mark.asyncio
async def test_claim_reflection_graph_runs_on_empty_state(fake_registry, fake_subjects_repo):
    graph = build_claim_reflection_graph(
        registry=fake_registry,
        subjects_repo=fake_subjects_repo,
    )
    initial = ReflectionState(session_id="test-session")
    result = await graph.ainvoke(initial)
    # End-to-end run should return a state-shaped dict without crashing.
    assert "candidate_subjects" in result
    assert "votes" in result
    assert result["candidate_subjects"] == []
    assert result["votes"] == []
    # Skeleton's persist_votes sets PROCESSING.
    assert "status" in result


def test_claim_reflection_graph_has_expected_nodes(fake_registry, fake_subjects_repo):
    graph = build_claim_reflection_graph(
        registry=fake_registry,
        subjects_repo=fake_subjects_repo,
    )
    nodes = set(graph.get_graph().nodes.keys())
    expected = {"route_candidates", "classify_stance", "propose_subject", "persist_votes"}
    assert expected.issubset(nodes)
