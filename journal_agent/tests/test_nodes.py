"""Layer 6 tests — Graph nodes (classifiers.py, stores.py).

Mocking strategy:
- Classifier nodes: stub LLMClient whose .structured()/.astructured()
  returns a mock runnable producing fixed Pydantic instances.
  Async nodes (thread_classifier, fragment_extractor) use .ainvoke().
- Save nodes: monkeypatch resolve_project_root → tmp_path for JsonlGateway;
  MagicMock for FragmentStore.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from journal_agent.comms.llm_client import LLMClient
from journal_agent.graph.nodes.classifiers import (
    inflate_threads,
    make_exchange_decomposer,
    make_intent_classifier,
    make_thread_classifier,
    make_thread_fragment_extractor,
)
from journal_agent.graph.nodes.stores import (
    make_save_classified_threads,
    make_save_fragments,
    make_save_threads,
    make_save_transcript,
)
from journal_agent.model.session import (
    ContextSpecification,
    Domain,
    Exchange,
    Fragment,
    FragmentDraft,
    FragmentDraftList,
    PromptKey,
    Role,
    ScoreCard,
    Status,
    Tag,
    ThreadClassificationResponse,
    ThreadSegment,
    ThreadSegmentList,
    Turn,
    UserProfile,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_exchange(session_id: str = "s1", human: str = "hello", ai: str = "hi") -> Exchange:
    return Exchange(
        session_id=session_id,
        human=Turn(session_id=session_id, role=Role.HUMAN, content=human),
        ai=Turn(session_id=session_id, role=Role.AI, content=ai),
    )


def _make_thread(exchange_ids: list[str], name: str = "topic_one") -> ThreadSegment:
    return ThreadSegment(
        thread_name=name,
        exchange_ids=exchange_ids,
        tags=[Tag(tag="philosophy")],
    )


def _make_fragment(session_id: str = "s1") -> Fragment:
    return Fragment(
        session_id=session_id,
        content="a thought",
        exchange_ids=["e1"],
        tags=[Tag(tag="philosophy")],
        timestamp=datetime.now(),
    )


def _stub_llm_client(return_value) -> LLMClient:
    """Create an LLMClient whose .structured(schema).invoke() returns *return_value*."""
    mock_chat = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = return_value
    mock_chat.with_structured_output.return_value = mock_runnable
    client = LLMClient(model="stub", client=mock_chat)
    return client


def _stub_async_llm_client(return_value) -> LLMClient:
    """Create an LLMClient whose .astructured(schema).ainvoke() returns *return_value*.

    Used for async nodes (thread_classifier, fragment_extractor) that call
    ``await structured_llm.ainvoke(...)``.
    """
    mock_chat = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=return_value)
    mock_chat.with_structured_output.return_value = mock_runnable
    client = LLMClient(model="stub", client=mock_chat)
    return client


def _base_state(**overrides) -> dict:
    """Minimal JournalState dict with sane defaults."""
    state = {
        "session_id": "test-session",
        "recent_messages": [],
        "session_messages": [],
        "transcript": [],
        "threads": [],
        "classified_threads": [],
        "fragments": [],
        "retrieved_history": [],
        "context_specification": ContextSpecification(),
        "user_profile": UserProfile(),
        "status": Status.IDLE,
        "error_message": None,
    }
    state.update(overrides)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# inflate_threads — pure helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestInflateThreads:
    def test_returns_expanded_threads_with_dialog(self):
        ex = _make_exchange()
        thread = _make_thread([ex.exchange_id])
        expanded = inflate_threads([thread], [ex])
        assert len(expanded) == 1
        assert expanded[0].thread_name == "topic_one"
        assert len(expanded[0].exchanges) == 1
        assert "hello" in expanded[0].exchanges[0].dialog

    def test_multiple_exchanges_sorted_by_timestamp(self):
        e1 = _make_exchange(human="first")
        e2 = _make_exchange(human="second")
        # ensure e2 is later
        e2.timestamp = datetime(2099, 1, 1)
        thread = _make_thread([e1.exchange_id, e2.exchange_id])
        expanded = inflate_threads([thread], [e1, e2])
        assert expanded[0].exchanges[-1].dialog.startswith("Human:\nsecond")

    def test_empty_threads_returns_empty(self):
        assert inflate_threads([], []) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier nodes
# ═══════════════════════════════════════════════════════════════════════════════

class TestExchangeDecomposer:
    def test_returns_threads_from_stubbed_llm(self):
        exchange = _make_exchange()
        thread = ThreadSegment(
            thread_name="test_thread",
            exchange_ids=[exchange.exchange_id],
            tags=[Tag(tag="general")],
        )
        llm = _stub_llm_client(ThreadSegmentList(threads=[thread]))
        node = make_exchange_decomposer(llm)
        result = node(_base_state(transcript=[exchange]))
        assert "threads" in result
        assert len(result["threads"]) == 1
        assert result["threads"][0].thread_name == "test_thread"

    def test_returns_error_on_llm_exception(self):
        mock_chat = MagicMock()
        mock_chat.with_structured_output.side_effect = RuntimeError("LLM down")
        llm = LLMClient(model="stub", client=mock_chat)
        node = make_exchange_decomposer(llm)
        result = node(_base_state(transcript=[_make_exchange()]))
        assert result["status"] == Status.ERROR
        assert "LLM down" in result["error_message"]


class TestThreadClassifier:
    @pytest.mark.asyncio
    async def test_returns_classified_threads_with_tags(self):
        exchange = _make_exchange()
        thread = _make_thread([exchange.exchange_id])
        tags_response = ThreadClassificationResponse(
            tags=[Tag(tag="creativity"), Tag(tag="philosophy")]
        )
        llm = _stub_async_llm_client(tags_response)
        node = make_thread_classifier(llm)
        result = await node(_base_state(transcript=[exchange], threads=[thread]))
        assert "classified_threads" in result
        assert len(result["classified_threads"]) == 1
        tag_names = [t.tag for t in result["classified_threads"][0].tags]
        assert "creativity" in tag_names

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_exception(self):
        exchange = _make_exchange()
        thread = _make_thread([exchange.exchange_id])
        mock_chat = MagicMock()
        mock_chat.with_structured_output.side_effect = RuntimeError("fail")
        llm = LLMClient(model="stub", client=mock_chat)
        node = make_thread_classifier(llm)
        result = await node(_base_state(transcript=[exchange], threads=[thread]))
        assert result["status"] == Status.ERROR


class TestFragmentExtractor:
    @pytest.mark.asyncio
    async def test_returns_fragments_from_drafts(self):
        exchange = _make_exchange()
        thread = _make_thread([exchange.exchange_id])
        drafts = FragmentDraftList(fragments=[
            FragmentDraft(
                content="an insight",
                exchange_ids=[exchange.exchange_id],
                tags=[Tag(tag="philosophy")],
            )
        ])
        llm = _stub_async_llm_client(drafts)
        node = make_thread_fragment_extractor(llm)
        result = await node(_base_state(
            transcript=[exchange],
            classified_threads=[thread],
        ))
        assert "fragments" in result
        assert len(result["fragments"]) == 1
        assert result["fragments"][0].content == "an insight"
        assert result["fragments"][0].session_id == "test-session"

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_exception(self):
        exchange = _make_exchange()
        thread = _make_thread([exchange.exchange_id])
        mock_chat = MagicMock()
        mock_chat.with_structured_output.side_effect = RuntimeError("boom")
        llm = LLMClient(model="stub", client=mock_chat)
        node = make_thread_fragment_extractor(llm)
        result = await node(_base_state(
            transcript=[exchange],
            classified_threads=[thread],
        ))
        assert result["status"] == Status.ERROR


class TestIntentClassifier:
    def test_returns_context_specification_via_score_card(self):
        score = ScoreCard(
            question_score=0.0,
            first_person_score=0.0,
            task_score=0.0,
            domains=[],
        )
        llm = _stub_llm_client(score)
        node = make_intent_classifier(llm)
        result = node(_base_state(
            session_messages=[HumanMessage(content="I wonder about life")],
        ))
        assert "context_specification" in result
        assert result["context_specification"].prompt_key == PromptKey.SOCRATIC

    def test_returns_error_when_no_session_messages(self):
        llm = _stub_llm_client(None)  # never reached
        node = make_intent_classifier(llm)
        result = node(_base_state(session_messages=[]))
        assert result["status"] == Status.ERROR
        assert "No session messages" in result["error_message"]

    def test_returns_error_on_llm_exception(self):
        mock_chat = MagicMock()
        mock_chat.with_structured_output.side_effect = RuntimeError("LLM fail")
        llm = LLMClient(model="stub", client=mock_chat)
        node = make_intent_classifier(llm)
        result = node(_base_state(
            session_messages=[HumanMessage(content="hi")],
        ))
        assert result["status"] == Status.ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# Save nodes
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveTranscript:
    def test_saves_transcript_to_mock_store(self):
        mock_store = MagicMock()
        exchange = _make_exchange()
        node = make_save_transcript(store=mock_store)
        result = node(_base_state(transcript=[exchange]))
        assert result["status"] == Status.TRANSCRIPT_SAVED
        mock_store.save_collection.assert_called_once()

    def test_returns_error_on_exception(self):
        broken_store = MagicMock()
        broken_store.save_collection.side_effect = RuntimeError("disk full")
        node = make_save_transcript(store=broken_store)
        result = node(_base_state(transcript=[_make_exchange()]))
        assert result["status"] == Status.ERROR


class TestSaveThreads:
    def test_saves_threads_to_mock_store(self):
        mock_store = MagicMock()
        thread = _make_thread(["e1"])
        node = make_save_threads(store=mock_store)
        result = node(_base_state(threads=[thread]))
        assert result["status"] == Status.THREADS_SAVED
        mock_store.save_collection.assert_called_once()


class TestSaveClassifiedThreads:
    def test_saves_classified_threads_to_mock_store(self):
        mock_store = MagicMock()
        thread = _make_thread(["e1"])
        node = make_save_classified_threads(store=mock_store)
        result = node(_base_state(classified_threads=[thread]))
        assert result["status"] == Status.CLASSIFIED_THREADS_SAVED
        mock_store.save_collection.assert_called_once()


class TestSaveFragments:
    def test_delegates_to_fragment_store(self):
        mock_store = MagicMock()
        fragment = _make_fragment()
        node = make_save_fragments(mock_store)
        result = node(_base_state(fragments=[fragment]))
        assert result["status"] == Status.FRAGMENTS_SAVED
        mock_store.save_fragments.assert_called_once_with([fragment])

    def test_returns_error_on_fragment_store_exception(self):
        mock_store = MagicMock()
        mock_store.save_fragments.side_effect = RuntimeError("store fail")
        node = make_save_fragments(mock_store)
        result = node(_base_state(fragments=[_make_fragment()]))
        assert result["status"] == Status.ERROR
        assert "store fail" in result["error_message"]
