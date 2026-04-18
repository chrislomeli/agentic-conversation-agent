"""Layer 7 tests — Storage: JsonStore, TranscriptStore, to_messages.

Mocking strategy:
- JsonStore / TranscriptStore: monkeypatch resolve_project_root → tmp_path

VectorStore tests live in test_vector_store.py (separate file because
vector_store.py imports chromadb at module level).
"""

import time
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from journal_agent.model.session import Exchange, Role, Turn
from journal_agent.storage.exchange_store import TranscriptStore, to_messages
from journal_agent.storage.storage import JsonStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_exchange(session_id: str = "s1") -> Exchange:
    return Exchange(
        session_id=session_id,
        human=Turn(session_id=session_id, role=Role.HUMAN, content="hello"),
        ai=Turn(session_id=session_id, role=Role.AI, content="response"),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def json_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "journal_agent.storage.storage.resolve_project_root",
        lambda: tmp_path,
    )
    return JsonStore("transcripts")


@pytest.fixture
def transcript_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "journal_agent.storage.storage.resolve_project_root",
        lambda: tmp_path,
    )
    return TranscriptStore()


# ═══════════════════════════════════════════════════════════════════════════════
# JsonStore — file I/O
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonStore:
    def test_save_then_load_round_trips_exchanges(self, json_store):
        exchange = make_exchange("sess-1")
        json_store.save_session("sess-1", [exchange])
        loaded = json_store.load_session("sess-1")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].session_id == "sess-1"
        assert loaded[0].human.content == "hello"
        assert loaded[0].ai.content == "response"

    def test_save_appends_to_existing_file(self, json_store):
        e1 = make_exchange("sess-a")
        e2 = make_exchange("sess-a")
        json_store.save_session("sess-a", [e1])
        json_store.save_session("sess-a", [e2])
        loaded = json_store.load_session("sess-a")
        assert len(loaded) == 2

    def test_load_returns_none_for_missing_session(self, json_store):
        result = json_store.load_session("no-such-session")
        assert result is None

    def test_load_returns_none_for_empty_file(self, json_store):
        # save_session with empty list writes nothing
        json_store.save_session("empty-sess", [])
        result = json_store.load_session("empty-sess")
        assert result is None

    def test_get_last_session_id_returns_none_for_empty_store(self, json_store):
        assert json_store.get_last_session_id() is None

    def test_get_last_session_id_returns_stem_of_newest_file(self, json_store):
        json_store.save_session("old-session", [make_exchange()])
        time.sleep(0.01)
        json_store.save_session("new-session", [make_exchange()])
        latest = json_store.get_last_session_id()
        assert latest == "new-session"

    def test_exchange_ids_survive_round_trip(self, json_store):
        exchange = make_exchange("s1")
        json_store.save_session("s1", [exchange])
        loaded = json_store.load_session("s1")
        assert loaded[0].exchange_id == exchange.exchange_id

    def test_save_creates_directory_if_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.storage.storage.resolve_project_root",
            lambda: tmp_path,
        )
        store = JsonStore("brand-new-folder")
        store.save_session("s1", [make_exchange()])
        assert (tmp_path / "data" / "brand-new-folder" / "s1.jsonl").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# to_messages — pure function
# ═══════════════════════════════════════════════════════════════════════════════

class TestToMessages:
    def test_human_turn_becomes_human_message(self):
        exchange = make_exchange()
        messages = to_messages([exchange])
        assert any(isinstance(m, HumanMessage) and m.content == "hello" for m in messages)

    def test_ai_turn_becomes_ai_message(self):
        exchange = make_exchange()
        messages = to_messages([exchange])
        assert any(isinstance(m, AIMessage) and m.content == "response" for m in messages)

    def test_order_is_human_then_ai_per_exchange(self):
        exchange = make_exchange()
        messages = to_messages([exchange])
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_multiple_exchanges_produces_interleaved_messages(self):
        exchanges = [make_exchange(), make_exchange()]
        messages = to_messages(exchanges)
        assert len(messages) == 4

    def test_empty_exchanges_returns_empty_list(self):
        assert to_messages([]) == []

    def test_system_role_becomes_system_message(self):
        exchange = Exchange(
            session_id="s1",
            human=Turn(session_id="s1", role=Role.SYSTEM, content="sys"),
            ai=Turn(session_id="s1", role=Role.AI, content="ok"),
        )
        messages = to_messages([exchange])
        assert isinstance(messages[0], SystemMessage)


# ═══════════════════════════════════════════════════════════════════════════════
# TranscriptStore — stateful turn accumulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTranscriptStore:
    def test_on_human_turn_sets_human_content(self, transcript_store):
        transcript_store.on_human_turn("s1", Role.HUMAN, "hello human")
        assert transcript_store._current_exchange.human.content == "hello human"

    def test_on_ai_turn_sets_ai_content_and_returns_exchange(self, transcript_store):
        transcript_store.on_human_turn("s1", Role.HUMAN, "hello")
        exchange = transcript_store.on_ai_turn("s1", Role.AI, "hello back")
        assert exchange.ai.content == "hello back"
        assert exchange.session_id == "s1"

    def test_on_ai_turn_resets_current_exchange(self, transcript_store):
        transcript_store.on_human_turn("s1", Role.HUMAN, "msg1")
        transcript_store.on_ai_turn("s1", Role.AI, "resp1")
        # current exchange is fresh after completing one pair
        assert transcript_store._current_exchange.human is None
        assert transcript_store._current_exchange.ai is None

    def test_on_ai_turn_timestamp_matches_ai_turn(self, transcript_store):
        transcript_store.on_human_turn("s1", Role.HUMAN, "q")
        exchange = transcript_store.on_ai_turn("s1", Role.AI, "a")
        assert exchange.timestamp == exchange.ai.timestamp

    def test_retrieve_transcript_returns_none_when_no_sessions_saved(self, transcript_store):
        result = transcript_store.retrieve_transcript()
        assert result is None

    def test_on_ai_turn_appends_to_exchanges_buffer(self, transcript_store):
        transcript_store.on_human_turn("s1", Role.HUMAN, "hello")
        transcript_store.on_ai_turn("s1", Role.AI, "hi")
        assert len(transcript_store._exchanges) == 1

    def test_store_cache_saves_exchanges_and_clears_buffer(self, transcript_store):
        transcript_store._exchanges.append(make_exchange())
        transcript_store.store_cache("s1")
        assert transcript_store._exchanges == []
        # verify it was actually persisted
        assert transcript_store._session_store.load_session("s1") is not None

    def test_store_cache_flushes_buffered_turns_to_disk(self, transcript_store):
        transcript_store.on_human_turn("s1", Role.HUMAN, "hello")
        transcript_store.on_ai_turn("s1", Role.AI, "hi")
        transcript_store.store_cache("s1")
        assert transcript_store._exchanges == []
        loaded = transcript_store._session_store.load_session("s1")
        assert loaded is not None
        assert len(loaded) == 1

