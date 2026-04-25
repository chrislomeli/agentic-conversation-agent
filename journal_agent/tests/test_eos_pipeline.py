"""Tests for the single end-of-session pipeline node (design plan #1).

What these tests prove:

    1. All 7 phases run in sequence when everything succeeds.
    2. State is threaded correctly between phases — a phase sees the outputs
       of all prior phases via model_copy, not just the original state.
    3. An error in one phase stops the pipeline immediately; no downstream
       phase runs.
    4. The accumulated state dict returned by the node is correct.
    5. ``_call`` works for both sync and async phase functions.

Strategy: mock each phase function individually, injecting canned return
dicts, and assert call order + state propagation without any LLM or DB I/O.
"""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from journal_agent.graph.nodes.eos_pipeline import _call, make_end_of_session_node
from journal_agent.graph.state import JournalState
from journal_agent.model.session import Exchange, StatusValue, ThreadSegment


# ── _call helper ─────────────────────────────────────────────────────────────


async def test_call_invokes_sync_function():
    def sync_fn(state):
        return {"from": "sync"}

    state = JournalState(session_id="s1")
    result = await _call(sync_fn, state)
    assert result == {"from": "sync"}


async def test_call_awaits_async_function():
    async def async_fn(state):
        return {"from": "async"}

    state = JournalState(session_id="s1")
    result = await _call(async_fn, state)
    assert result == {"from": "async"}


# ── make_end_of_session_node: happy path ─────────────────────────────────────


def _make_dummy_exchange():
    from journal_agent.model.session import Turn, Role
    human_turn = Turn(session_id="s1", role=Role.HUMAN, content="hi")
    ai_turn = Turn(session_id="s1", role=Role.AI, content="hello")
    return Exchange(human=human_turn, ai=ai_turn)


def _state_with_transcript() -> JournalState:
    """JournalState with one exchange in the transcript (needed by EOS phases)."""
    exchange = _make_dummy_exchange()
    return JournalState(session_id="s1", transcript=[exchange])


def _mock_deps():
    """Return keyword-argument mock dependencies for make_end_of_session_node."""
    return {
        "transcript_store": MagicMock(),
        "thread_store": MagicMock(),
        "classified_thread_store": MagicMock(),
        "fragment_store": MagicMock(),
        "classifier_llm": MagicMock(),
        "extractor_llm": MagicMock(),
    }


async def test_all_phases_run_when_no_errors():
    """Seven phase calls must fire in the right order on a clean pipeline run."""
    call_order = []

    def _phase(name, returns=None):
        def _sync(state):
            call_order.append(name)
            return returns or {}
        return _sync

    thread = ThreadSegment(thread_name="t1", exchange_ids=["e1"], tags=[])

    with (
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_transcript",
              return_value=_phase("save_transcript", {"status": StatusValue.TRANSCRIPT_SAVED})),
        patch("journal_agent.graph.nodes.eos_pipeline.make_exchange_decomposer",
              return_value=_phase("exchange_decomposer", {"threads": [thread]})),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_threads",
              return_value=_phase("save_threads", {"status": StatusValue.THREADS_SAVED})),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_classifier",
              return_value=_phase("thread_classifier", {"classified_threads": [thread]})),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_classified_threads",
              return_value=_phase("save_classified_threads", {"status": StatusValue.CLASSIFIED_THREADS_SAVED})),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_fragment_extractor",
              return_value=_phase("thread_fragment_extractor", {"fragments": []})),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_fragments",
              return_value=_phase("save_fragments", {"status": StatusValue.FRAGMENTS_SAVED})),
    ):
        node = make_end_of_session_node(**_mock_deps())
        await node(_state_with_transcript())

    assert call_order == [
        "save_transcript",
        "exchange_decomposer",
        "save_threads",
        "thread_classifier",
        "save_classified_threads",
        "thread_fragment_extractor",
        "save_fragments",
    ]


async def test_state_threads_forward_between_phases():
    """A phase sees the output of all prior phases, not just the original state."""
    seen_threads: list = []

    thread = ThreadSegment(thread_name="t1", exchange_ids=["e1"], tags=[])

    def _decomposer(state):
        return {"threads": [thread]}

    def _classifier(state):
        # thread_classifier runs AFTER decomposer; it must see state.threads
        seen_threads.extend(state.threads)
        return {"classified_threads": [thread]}

    with (
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_transcript",
              return_value=lambda s: {"status": StatusValue.TRANSCRIPT_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_exchange_decomposer",
              return_value=_decomposer),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_threads",
              return_value=lambda s: {"status": StatusValue.THREADS_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_classifier",
              return_value=_classifier),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_classified_threads",
              return_value=lambda s: {"status": StatusValue.CLASSIFIED_THREADS_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_fragment_extractor",
              return_value=lambda s: {"fragments": []}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_fragments",
              return_value=lambda s: {"status": StatusValue.FRAGMENTS_SAVED}),
    ):
        node = make_end_of_session_node(**_mock_deps())
        await node(_state_with_transcript())

    assert len(seen_threads) == 1, "thread_classifier must see threads from exchange_decomposer"
    assert seen_threads[0].thread_name == "t1"


async def test_error_in_phase_stops_pipeline():
    """A STATUS.ERROR from any phase must halt the pipeline immediately."""
    call_order = []

    def _save_transcript(state):
        call_order.append("save_transcript")
        return {"status": StatusValue.ERROR, "error_message": "disk full"}

    def _decomposer(state):
        call_order.append("exchange_decomposer")  # must NOT be called
        return {"threads": []}

    with (
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_transcript",
              return_value=_save_transcript),
        patch("journal_agent.graph.nodes.eos_pipeline.make_exchange_decomposer",
              return_value=_decomposer),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_threads",
              return_value=lambda s: {}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_classifier",
              return_value=lambda s: {}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_classified_threads",
              return_value=lambda s: {}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_fragment_extractor",
              return_value=lambda s: {}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_fragments",
              return_value=lambda s: {}),
    ):
        node = make_end_of_session_node(**_mock_deps())
        result = await node(_state_with_transcript())

    assert result["status"] == StatusValue.ERROR
    assert result["error_message"] == "disk full"
    assert call_order == ["save_transcript"], "Pipeline must stop after first error"


async def test_accumulated_dict_contains_all_phase_outputs():
    """The final return dict must include updates from all successful phases."""
    thread = ThreadSegment(thread_name="t1", exchange_ids=["e1"], tags=[])

    with (
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_transcript",
              return_value=lambda s: {"status": StatusValue.TRANSCRIPT_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_exchange_decomposer",
              return_value=lambda s: {"threads": [thread]}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_threads",
              return_value=lambda s: {"status": StatusValue.THREADS_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_classifier",
              return_value=lambda s: {"classified_threads": [thread]}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_classified_threads",
              return_value=lambda s: {"status": StatusValue.CLASSIFIED_THREADS_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_fragment_extractor",
              return_value=lambda s: {"fragments": []}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_fragments",
              return_value=lambda s: {"status": StatusValue.FRAGMENTS_SAVED}),
    ):
        node = make_end_of_session_node(**_mock_deps())
        result = await node(_state_with_transcript())

    assert result["threads"] == [thread]
    assert result["classified_threads"] == [thread]
    assert result["fragments"] == []
    assert result["status"] == StatusValue.FRAGMENTS_SAVED


async def test_pipeline_works_with_async_phases():
    """Async phase functions (thread_classifier, thread_fragment_extractor) must work."""

    async def _async_classifier(state):
        return {"classified_threads": []}

    async def _async_extractor(state):
        return {"fragments": []}

    with (
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_transcript",
              return_value=lambda s: {"status": StatusValue.TRANSCRIPT_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_exchange_decomposer",
              return_value=lambda s: {"threads": []}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_threads",
              return_value=lambda s: {"status": StatusValue.THREADS_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_classifier",
              return_value=_async_classifier),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_classified_threads",
              return_value=lambda s: {"status": StatusValue.CLASSIFIED_THREADS_SAVED}),
        patch("journal_agent.graph.nodes.eos_pipeline.make_thread_fragment_extractor",
              return_value=_async_extractor),
        patch("journal_agent.graph.nodes.eos_pipeline.make_save_fragments",
              return_value=lambda s: {"status": StatusValue.FRAGMENTS_SAVED}),
    ):
        node = make_end_of_session_node(**_mock_deps())
        result = await node(_state_with_transcript())

    # No assertion needed beyond "it didn't raise" — proves async phases execute.
    assert "fragments" in result
