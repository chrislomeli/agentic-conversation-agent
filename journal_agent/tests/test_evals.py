"""Tests for the eval harness (#7-hardening).

All tests are offline — no LLM calls.  The LLM is mocked to return
canned structured outputs so we can verify the harness mechanics:
fixture loading, state construction, record serialisation, EOS chaining,
and the comparator.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from journal_agent.evals.fixtures import (
    FIXTURE_PATH,
    Fixture,
    build_eos_state,
    build_intent_state,
    input_hash,
    load_fixtures,
)
from journal_agent.evals.runner import EvalRecord, load_results, run_suite, write_results
from journal_agent.evals.compare import compare_runs
from journal_agent.model.session import (
    ContextSpecification,
    Exchange,
    Fragment,
    FragmentDraft,
    FragmentDraftList,
    PromptKey,
    ScoreCard,
    Tag,
    ThreadClassificationResponse,
    ThreadSegment,
    ThreadSegmentList,
    Turn,
    Role,
)

from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Fixture loading
# ═══════════════════════════════════════════════════════════════════════════════


def test_builtin_fixture_file_exists():
    assert FIXTURE_PATH.exists(), f"Built-in fixture file missing: {FIXTURE_PATH}"


def test_load_fixtures_returns_list():
    fixtures = load_fixtures()
    assert isinstance(fixtures, list)
    assert len(fixtures) > 0


def test_each_fixture_has_required_fields():
    for f in load_fixtures():
        assert f.fixture_id, "fixture_id must be non-empty"
        assert f.description, "description must be non-empty"
        assert len(f.exchanges) >= 1, "fixture must have at least one exchange"


def test_fixture_exchanges_are_exchange_objects():
    fixtures = load_fixtures()
    for f in fixtures:
        for ex in f.exchanges:
            assert isinstance(ex, Exchange)


def test_fixture_ids_are_unique():
    fixtures = load_fixtures()
    ids = [f.fixture_id for f in fixtures]
    assert len(ids) == len(set(ids)), "fixture_id values must be unique"


def test_load_fixtures_custom_path(tmp_path):
    fixture_file = tmp_path / "test.jsonl"
    exchange_data = {
        "exchange_id": "e1",
        "session_id": "f_test",
        "timestamp": "2026-01-01T00:00:00",
        "human": {"session_id": "f_test", "role": "human", "content": "hello", "timestamp": "2026-01-01T00:00:00"},
        "ai": {"session_id": "f_test", "role": "ai", "content": "hi there", "timestamp": "2026-01-01T00:00:10"},
    }
    fixture_data = {"fixture_id": "custom_f01", "description": "test", "exchanges": [exchange_data]}
    fixture_file.write_text(json.dumps(fixture_data) + "\n")

    fixtures = load_fixtures(fixture_file)
    assert len(fixtures) == 1
    assert fixtures[0].fixture_id == "custom_f01"
    assert len(fixtures[0].exchanges) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: State construction
# ═══════════════════════════════════════════════════════════════════════════════


def _make_fixture(fixture_id: str = "t001") -> Fixture:
    ex = Exchange(
        exchange_id="e1",
        session_id=fixture_id,
        timestamp=datetime(2026, 1, 1),
        human=Turn(session_id=fixture_id, role=Role.HUMAN, content="I feel anxious about my job."),
        ai=Turn(session_id=fixture_id, role=Role.AI, content="Tell me more about what's worrying you."),
    )
    return Fixture(fixture_id=fixture_id, description="test", exchanges=[ex])


def test_build_intent_state_has_session_messages():
    from langchain_core.messages import HumanMessage, AIMessage
    fixture = _make_fixture()
    state = build_intent_state(fixture)
    assert len(state.session_messages) == 2
    assert isinstance(state.session_messages[0], HumanMessage)
    assert isinstance(state.session_messages[1], AIMessage)


def test_build_intent_state_session_id_matches_fixture():
    fixture = _make_fixture("ftest")
    state = build_intent_state(fixture)
    assert state.session_id == "ftest"


def test_build_eos_state_has_transcript():
    fixture = _make_fixture()
    state = build_eos_state(fixture)
    assert len(state.transcript) == 1
    assert state.transcript[0].exchange_id == "e1"


def test_build_eos_state_threads_empty_initially():
    fixture = _make_fixture()
    state = build_eos_state(fixture)
    assert state.threads == []
    assert state.classified_threads == []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: input_hash stability
# ═══════════════════════════════════════════════════════════════════════════════


def test_input_hash_is_deterministic():
    fixture = _make_fixture()
    state = build_intent_state(fixture)
    h1 = input_hash(state.session_messages)
    h2 = input_hash(state.session_messages)
    assert h1 == h2


def test_input_hash_changes_on_different_content():
    from langchain_core.messages import HumanMessage
    h1 = input_hash([HumanMessage(content="hello")])
    h2 = input_hash([HumanMessage(content="goodbye")])
    assert h1 != h2


def test_input_hash_is_eight_chars():
    from langchain_core.messages import HumanMessage
    h = input_hash([HumanMessage(content="test")])
    assert len(h) == 8


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EvalRecord serialisation
# ═══════════════════════════════════════════════════════════════════════════════


def _make_record(**kwargs) -> EvalRecord:
    defaults = dict(
        fixture_id="f001",
        classifier="intent_classifier",
        prompt_key="intent_classifier",
        prompt_version="v1",
        input_hash="abcd1234",
        output={"prompt_key": "conversation"},
        elapsed_ms=500,
        timestamp="2026-01-01T00:00:00+00:00",
        error=None,
    )
    defaults.update(kwargs)
    return EvalRecord(**defaults)


def test_eval_record_serialises_to_json():
    record = _make_record()
    raw = record.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["fixture_id"] == "f001"
    assert parsed["classifier"] == "intent_classifier"


def test_write_and_load_results_roundtrip(tmp_path):
    records = [_make_record(), _make_record(fixture_id="f002", classifier="exchange_decomposer")]
    out = tmp_path / "results.jsonl"
    write_results(records, out)
    loaded = load_results(out)
    assert len(loaded) == 2
    assert loaded[0].fixture_id == "f001"
    assert loaded[1].fixture_id == "f002"


def test_write_results_creates_parent_dirs(tmp_path):
    records = [_make_record()]
    out = tmp_path / "nested" / "dir" / "results.jsonl"
    write_results(records, out)
    assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: run_suite (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_mock_llm() -> MagicMock:
    """Return a mock LLMClient that returns canned structured outputs."""
    llm = MagicMock()

    score_card = ScoreCard(question_score=0.2, first_person_score=0.8,
                           personalization_score=0.0, task_score=0.1)
    thread_list = ThreadSegmentList(threads=[
        ThreadSegment(thread_name="work_stress", exchange_ids=["e1"], tags=[])
    ])
    classification = ThreadClassificationResponse(tags=[Tag(tag="emotions_mental_health")])
    fragments = FragmentDraftList(fragments=[
        FragmentDraft(content="User is anxious about job", exchange_ids=["e1"], tags=[])
    ])

    # sync structured (intent_classifier, exchange_decomposer)
    llm.structured.return_value.invoke.side_effect = [score_card, thread_list]

    # async structured (thread_classifier, fragment_extractor)
    llm.astructured.return_value.ainvoke = AsyncMock(side_effect=[classification, fragments])

    return llm


def test_run_suite_produces_four_records_per_fixture():
    fixture = _make_fixture("f001")
    llm = _build_mock_llm()

    with patch("journal_agent.graph.nodes.classifiers.get_prompt", return_value="prompt text"), \
         patch("journal_agent.configure.context_builder.ContextBuilder.get_context", return_value=[]):
        records = asyncio.get_event_loop().run_until_complete(run_suite([fixture], llm))

    assert len(records) == 4
    classifiers = [r.classifier for r in records]
    assert "intent_classifier" in classifiers
    assert "exchange_decomposer" in classifiers
    assert "thread_classifier" in classifiers
    assert "thread_fragment_extractor" in classifiers


def test_run_suite_records_have_prompt_version():
    fixture = _make_fixture("f001")
    llm = _build_mock_llm()

    with patch("journal_agent.graph.nodes.classifiers.get_prompt", return_value="prompt"), \
         patch("journal_agent.configure.context_builder.ContextBuilder.get_context", return_value=[]):
        records = asyncio.get_event_loop().run_until_complete(run_suite([fixture], llm))

    for r in records:
        assert r.prompt_version.startswith("v"), f"{r.classifier} missing prompt_version"


def test_run_suite_records_have_input_hash():
    fixture = _make_fixture("f001")
    llm = _build_mock_llm()

    with patch("journal_agent.graph.nodes.classifiers.get_prompt", return_value="prompt"), \
         patch("journal_agent.configure.context_builder.ContextBuilder.get_context", return_value=[]):
        records = asyncio.get_event_loop().run_until_complete(run_suite([fixture], llm))

    for r in records:
        assert len(r.input_hash) == 8, f"{r.classifier} input_hash wrong length"


def test_run_suite_chains_decomposer_output_to_classifier():
    """Thread count from decomposer must match what classifier receives."""
    fixture = _make_fixture("f001")
    llm = _build_mock_llm()

    with patch("journal_agent.graph.nodes.classifiers.get_prompt", return_value="prompt"), \
         patch("journal_agent.configure.context_builder.ContextBuilder.get_context", return_value=[]):
        records = asyncio.get_event_loop().run_until_complete(run_suite([fixture], llm))

    decomposer_rec = next(r for r in records if r.classifier == "exchange_decomposer")
    classifier_rec = next(r for r in records if r.classifier == "thread_classifier")

    thread_count_decomposed = len(decomposer_rec.output.get("threads", []))
    thread_count_classified = len(classifier_rec.output.get("classified_threads", []))
    # classifier runs once per thread from decomposer
    assert thread_count_classified == thread_count_decomposed


def test_run_suite_error_captured_in_record():
    fixture = _make_fixture("f001")
    llm = MagicMock()
    llm.structured.return_value.invoke.side_effect = RuntimeError("model timeout")
    # Provide fallback returns for subsequent classifiers
    llm.astructured.return_value.ainvoke = AsyncMock(return_value=ThreadClassificationResponse(tags=[]))

    with patch("journal_agent.graph.nodes.classifiers.get_prompt", return_value="prompt"), \
         patch("journal_agent.configure.context_builder.ContextBuilder.get_context", return_value=[]):
        records = asyncio.get_event_loop().run_until_complete(run_suite([fixture], llm))

    intent_rec = next(r for r in records if r.classifier == "intent_classifier")
    assert intent_rec.error is not None
    assert "timeout" in intent_rec.error


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Comparator
# ═══════════════════════════════════════════════════════════════════════════════


def _write_run(records: list[EvalRecord], tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    write_results(records, path)
    return path


def test_compare_same_runs_reports_same(tmp_path):
    records = [_make_record(), _make_record(fixture_id="f002", classifier="exchange_decomposer")]
    path_a = _write_run(records, tmp_path, "a.jsonl")
    path_b = _write_run(records, tmp_path, "b.jsonl")
    report = compare_runs(path_a, path_b)
    assert "2 same" in report


def test_compare_changed_output_reports_changed(tmp_path):
    a = [_make_record(output={"prompt_key": "conversation"})]
    b = [_make_record(output={"prompt_key": "socratic"})]
    path_a = _write_run(a, tmp_path, "a.jsonl")
    path_b = _write_run(b, tmp_path, "b.jsonl")
    report = compare_runs(path_a, path_b)
    assert "CHANGED" in report


def test_compare_changed_prompt_version_reports_changed(tmp_path):
    a = [_make_record(prompt_version="v1")]
    b = [_make_record(prompt_version="v2")]
    path_a = _write_run(a, tmp_path, "a.jsonl")
    path_b = _write_run(b, tmp_path, "b.jsonl")
    report = compare_runs(path_a, path_b)
    assert "CHANGED" in report
    assert "v1" in report
    assert "v2" in report


def test_compare_new_record_in_b(tmp_path):
    a = [_make_record()]
    b = [_make_record(), _make_record(fixture_id="f002", classifier="exchange_decomposer")]
    path_a = _write_run(a, tmp_path, "a.jsonl")
    path_b = _write_run(b, tmp_path, "b.jsonl")
    report = compare_runs(path_a, path_b)
    assert "NEW" in report


def test_compare_dropped_record(tmp_path):
    a = [_make_record(), _make_record(fixture_id="f002", classifier="exchange_decomposer")]
    b = [_make_record()]
    path_a = _write_run(a, tmp_path, "a.jsonl")
    path_b = _write_run(b, tmp_path, "b.jsonl")
    report = compare_runs(path_a, path_b)
    assert "DROPPED" in report


def test_compare_summary_line_present(tmp_path):
    records = [_make_record()]
    path = _write_run(records, tmp_path, "r.jsonl")
    report = compare_runs(path, path)
    assert "Summary:" in report
