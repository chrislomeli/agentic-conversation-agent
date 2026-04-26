"""Tests for design plan #8-design — observability.

Verifies three properties:

1. node_trace is always active — no env-var gate; wrapping a node always
   produces timing + status log records.

2. Structured log fields — every log record produced by node_trace and
   classifier decision-point logs carries the expected ``extra`` keys so
   log aggregators can filter without parsing the message string.

3. Decision-point coverage — intent_classifier, exchange_decomposer,
   thread_classifier, and thread_fragment_extractor each emit a decision log
   with the fields an eval harness needs.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import JournalState

_NODE_TRACER_LOGGER = "journal_agent.graph.node_tracer"
_CLASSIFIERS_LOGGER = "journal_agent.graph.nodes.classifiers"


@pytest.fixture(autouse=True)
def restore_node_tracer_propagation():
    """config_builder sets propagate=False so node_tracer goes to a file, not console.

    When test_config.py runs first it permanently mutates the logger in-process,
    causing caplog to miss node_tracer records.  Reset propagation for each
    observability test so caplog works regardless of suite ordering.
    """
    nt = logging.getLogger(_NODE_TRACER_LOGGER)
    cl = logging.getLogger(_CLASSIFIERS_LOGGER)
    orig_nt, orig_cl = nt.propagate, cl.propagate
    nt.propagate = True
    cl.propagate = True
    yield
    nt.propagate = orig_nt
    cl.propagate = orig_cl


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: node_trace is always-on
# ═══════════════════════════════════════════════════════════════════════════════


def _make_state(session_id: str = "test-session") -> JournalState:
    return JournalState(session_id=session_id)


def test_node_trace_wraps_sync_node(caplog):
    """node_trace wraps sync functions regardless of env vars."""

    @node_trace("my_node")
    def dummy_node(state: JournalState) -> dict:
        return {"status": "ok"}

    state = _make_state()
    with caplog.at_level(logging.INFO, logger="journal_agent.graph.node_tracer"):
        result = dummy_node(state)

    assert result == {"status": "ok"}
    assert any("node completed" in r.message for r in caplog.records)


def test_node_trace_wraps_async_node(caplog):
    """node_trace wraps async functions regardless of env vars."""
    import asyncio

    @node_trace("async_node")
    async def dummy_async_node(state: JournalState) -> dict:
        return {"status": "ok"}

    state = _make_state()
    with caplog.at_level(logging.INFO, logger="journal_agent.graph.node_tracer"):
        result = asyncio.get_event_loop().run_until_complete(dummy_async_node(state))

    assert result == {"status": "ok"}
    assert any("node completed" in r.message for r in caplog.records)


def test_node_trace_wraps_without_env_var(caplog, monkeypatch):
    """Removing NODE_TRACE_ENABLED from the environment does not disable tracing."""
    monkeypatch.delenv("NODE_TRACE_ENABLED", raising=False)

    @node_trace("env_test_node")
    def env_test_node(state: JournalState) -> dict:
        return {}

    state = _make_state()
    with caplog.at_level(logging.INFO, logger="journal_agent.graph.node_tracer"):
        env_test_node(state)

    assert any("node completed" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Structured log fields on node_trace records
# ═══════════════════════════════════════════════════════════════════════════════


def test_node_trace_log_has_node_field(caplog):
    @node_trace("my_node")
    def node(state: JournalState) -> dict:
        return {}

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.node_tracer"):
        node(_make_state())

    record = next(r for r in caplog.records if "node completed" in r.message)
    assert hasattr(record, "node"), "Log record missing 'node' field"
    assert record.node == "my_node"


def test_node_trace_log_has_session_id_field(caplog):
    @node_trace("sid_node")
    def node(state: JournalState) -> dict:
        return {}

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.node_tracer"):
        node(_make_state(session_id="sess-42"))

    record = next(r for r in caplog.records if "node completed" in r.message)
    assert hasattr(record, "session_id")
    assert record.session_id == "sess-42"


def test_node_trace_log_has_elapsed_ms_field(caplog):
    @node_trace("timed_node")
    def node(state: JournalState) -> dict:
        return {}

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.node_tracer"):
        node(_make_state())

    record = next(r for r in caplog.records if "node completed" in r.message)
    assert hasattr(record, "elapsed_ms")
    assert isinstance(record.elapsed_ms, int)
    assert record.elapsed_ms >= 0


def test_node_trace_error_node_logs_warning(caplog):
    from journal_agent.model.session import StatusValue

    @node_trace("err_node")
    def err_node(state: JournalState) -> dict:
        return {"status": StatusValue.ERROR, "error_message": "boom"}

    with caplog.at_level(logging.WARNING, logger="journal_agent.graph.node_tracer"):
        err_node(_make_state())

    record = next(r for r in caplog.records if "node completed with error" in r.message)
    assert record.levelno == logging.WARNING
    assert record.error_message == "boom"


def test_node_trace_exception_logs_at_exception_level(caplog):
    @node_trace("exc_node")
    def exc_node(state: JournalState) -> dict:
        raise RuntimeError("unexpected")

    with caplog.at_level(logging.ERROR, logger="journal_agent.graph.node_tracer"):
        with pytest.raises(RuntimeError):
            exc_node(_make_state())

    assert any("node raised exception" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Decision-point logs in classifiers
# ═══════════════════════════════════════════════════════════════════════════════


def test_intent_classifier_logs_decision(caplog):
    from unittest.mock import MagicMock
    from journal_agent.graph.nodes.classifiers import make_intent_classifier
    from journal_agent.model.session import ScoreCard, ContextSpecification, PromptKey

    score_card = ScoreCard(question_score=0.8, first_person_score=0.2,
                           personalization_score=0.1, task_score=0.3)
    specification = ContextSpecification(prompt_key=PromptKey.CONVERSATION)

    mock_llm = MagicMock()
    mock_llm.structured.return_value.invoke.return_value = score_card

    mock_cb = MagicMock()
    mock_cb.get_context.return_value = []

    from unittest.mock import patch
    with patch("journal_agent.graph.nodes.classifiers.resolve_scorecard_to_specification",
               return_value=specification):
        classifier = make_intent_classifier(mock_llm, context_builder=mock_cb)

    state = _make_state()
    state.session_messages = [MagicMock()]

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.nodes.classifiers"):
        classifier(state)

    record = next(
        (r for r in caplog.records if "intent_classifier decision" in r.message), None
    )
    assert record is not None, "No intent_classifier decision log found"
    assert hasattr(record, "question_score")
    assert hasattr(record, "personalization_score")
    assert hasattr(record, "prompt_key")


def test_exchange_decomposer_logs_thread_count(caplog):
    from journal_agent.graph.nodes.classifiers import make_exchange_decomposer
    from journal_agent.model.session import ThreadSegmentList, ThreadSegment

    thread_list = ThreadSegmentList(threads=[
        ThreadSegment(thread_name="t1", exchange_ids=[], tags=[]),
        ThreadSegment(thread_name="t2", exchange_ids=[], tags=[]),
    ])

    mock_llm = MagicMock()
    mock_llm.structured.return_value.invoke.return_value = thread_list

    decomposer = make_exchange_decomposer(mock_llm)
    state = _make_state()

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.nodes.classifiers"):
        decomposer(state)

    record = next(
        (r for r in caplog.records if "exchange_decomposer decision" in r.message), None
    )
    assert record is not None, "No exchange_decomposer decision log found"
    assert hasattr(record, "thread_count")
    assert record.thread_count == 2


def test_thread_classifier_logs_thread_count(caplog):
    import asyncio
    from journal_agent.graph.nodes.classifiers import make_thread_classifier
    from journal_agent.model.session import (
        ThreadSegment, Exchange, ThreadClassificationResponse
    )

    mock_response = ThreadClassificationResponse(tags=[])
    mock_llm = MagicMock()
    mock_llm.astructured.return_value.ainvoke = AsyncMock(return_value=mock_response)

    classifier = make_thread_classifier(mock_llm)

    state = _make_state()
    thread = ThreadSegment(thread_name="t1", exchange_ids=["e1"], tags=[])
    state.threads = [thread]
    exchange = Exchange()
    exchange.exchange_id = "e1"
    from unittest.mock import MagicMock as MM
    exchange.human = MM(content="hi")
    exchange.ai = MM(content="hello")
    state.transcript = [exchange]

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.nodes.classifiers"):
        asyncio.get_event_loop().run_until_complete(classifier(state))

    record = next(
        (r for r in caplog.records if "thread_classifier decision" in r.message), None
    )
    assert record is not None, "No thread_classifier decision log found"
    assert hasattr(record, "thread_count")


def test_fragment_extractor_logs_fragment_count(caplog):
    import asyncio
    from journal_agent.graph.nodes.classifiers import make_thread_fragment_extractor
    from journal_agent.model.session import (
        ThreadSegment, Exchange, FragmentDraftList, FragmentDraft
    )

    from datetime import datetime
    draft = FragmentDraft(content="insight", exchange_ids=[], tags=[])
    mock_response = FragmentDraftList(fragments=[draft])
    mock_llm = MagicMock()
    mock_llm.astructured.return_value.ainvoke = AsyncMock(return_value=mock_response)

    extractor = make_thread_fragment_extractor(mock_llm)

    state = _make_state()
    thread = ThreadSegment(thread_name="t1", exchange_ids=["e1"], tags=[])
    state.classified_threads = [thread]
    exchange = Exchange()
    exchange.exchange_id = "e1"
    from unittest.mock import MagicMock as MM
    exchange.human = MM(content="hi")
    exchange.ai = MM(content="hello")
    state.transcript = [exchange]

    with caplog.at_level(logging.INFO, logger="journal_agent.graph.nodes.classifiers"):
        asyncio.get_event_loop().run_until_complete(extractor(state))

    record = next(
        (r for r in caplog.records if "thread_fragment_extractor decision" in r.message), None
    )
    assert record is not None, "No thread_fragment_extractor decision log found"
    assert hasattr(record, "fragment_count")
    assert record.fragment_count == 1
