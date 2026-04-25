"""Tests for the FastAPI layer — items #5 and the SSE streaming helpers.

Strategy
--------
The lifespan wires Postgres, LLM registry, and compiled LangGraph graphs —
none of which we want in unit tests.  Instead each test fixture populates
``app.state`` directly with mocks, then sends requests via ``httpx.AsyncClient``
using ``ASGITransport(app=app)``.  This gives us full HTTP semantics (status
codes, headers, streaming) without any real I/O.

Coverage
--------
1. ``format_sse`` — correct SSE wire format.
2. ``graph_stream`` — token events, system_message pickup, done event, error path.
3. ``POST /sessions`` — creates a uuid session_id, marks it as new.
4. ``POST /chat/{session_id}`` — real graph invocation path, SSE stream returned.
5. ``POST /chat/{session_id}`` (first turn) — bootstrap fields injected once.
6. ``POST /chat/{session_id}`` (/quit) — returns system + done without graph call.
7. ``DELETE /sessions/{session_id}`` — EOS pipeline invoked, 202 returned.
8. ``GET /health`` — liveness probe.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk

from journal_agent.api.main import app
from journal_agent.api.models import SseEvent
from journal_agent.api.streaming import format_sse, graph_stream


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: format_sse — wire format
# ═══════════════════════════════════════════════════════════════════════════════


def test_format_sse_token_event():
    wire = format_sse(SseEvent.TOKEN, "Hello ")
    assert wire == 'event: token\ndata: {"text": "Hello "}\n\n'


def test_format_sse_done_event():
    wire = format_sse(SseEvent.DONE, "")
    assert wire == 'event: done\ndata: {"text": ""}\n\n'


def test_format_sse_system_event():
    wire = format_sse(SseEvent.SYSTEM, "Saved 3 exchanges.")
    assert wire.startswith("event: system\n")
    assert "Saved 3 exchanges." in wire


def test_format_sse_error_event():
    wire = format_sse(SseEvent.ERROR, "something broke")
    assert wire.startswith("event: error\n")
    assert "something broke" in wire


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: graph_stream — async generator behavior
# ═══════════════════════════════════════════════════════════════════════════════


def _chat_event(node: str, content: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": AIMessageChunk(content=content)},
    }


def _other_event(name: str) -> dict:
    return {"event": name, "metadata": {"langgraph_node": "get_ai_response"}, "data": {}}


async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen) -> list[dict]:
    """Collect all SSE events from an async generator into parsed dicts."""
    results = []
    async for line in gen:
        if line.startswith("event:"):
            parts = line.strip().split("\n")
            event_type = parts[0].split(": ", 1)[1]
            data = json.loads(parts[1].split(": ", 1)[1])
            results.append({"event": event_type, "data": data})
    return results


async def test_graph_stream_emits_tokens_from_ai_response_node():
    """Chunks from get_ai_response arrive as token SSE events."""
    events = _aiter(
        _chat_event("get_ai_response", "hello "),
        _chat_event("get_ai_response", "world"),
    )

    conv = MagicMock()
    conv.aget_state = AsyncMock(return_value=MagicMock(values={}))

    collected = await _collect(graph_stream(events, conv, {}))
    token_events = [e for e in collected if e["event"] == "token"]

    assert len(token_events) == 2
    assert token_events[0]["data"]["text"] == "hello "
    assert token_events[1]["data"]["text"] == "world"


async def test_graph_stream_filters_non_ai_response_nodes():
    """Tokens from classifier / extractor nodes must not appear in the stream."""
    events = _aiter(
        _chat_event("intent_classifier", "secret"),
        _chat_event("get_ai_response", "visible"),
    )

    conv = MagicMock()
    conv.aget_state = AsyncMock(return_value=MagicMock(values={}))

    collected = await _collect(graph_stream(events, conv, {}))
    token_events = [e for e in collected if e["event"] == "token"]

    assert len(token_events) == 1
    assert token_events[0]["data"]["text"] == "visible"


async def test_graph_stream_always_ends_with_done():
    """Every stream — even an empty one — must end with a done event."""
    conv = MagicMock()
    conv.aget_state = AsyncMock(return_value=MagicMock(values={}))

    collected = await _collect(graph_stream(_aiter(), conv, {}))

    assert collected[-1]["event"] == "done"


async def test_graph_stream_emits_system_message_from_state():
    """system_message left in graph state is forwarded as a system SSE event."""
    events = _aiter()
    conv = MagicMock()
    conv.aget_state = AsyncMock(
        return_value=MagicMock(values={"system_message": "Saved 2 exchanges."})
    )

    collected = await _collect(graph_stream(events, conv, {}))

    system_events = [e for e in collected if e["event"] == "system"]
    assert len(system_events) == 1
    assert system_events[0]["data"]["text"] == "Saved 2 exchanges."


async def test_graph_stream_emits_error_on_exception():
    """An exception inside the graph produces an error SSE event, not a crash."""
    async def _bad_iter():
        yield _chat_event("get_ai_response", "start")
        raise RuntimeError("graph error")

    conv = MagicMock()
    conv.aget_state = AsyncMock(return_value=MagicMock(values={}))

    collected = await _collect(graph_stream(_bad_iter(), conv, {}))

    error_events = [e for e in collected if e["event"] == "error"]
    assert len(error_events) == 1
    assert "graph error" in error_events[0]["data"]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HTTP endpoints — inject app.state, bypass lifespan
# ═══════════════════════════════════════════════════════════════════════════════


def _make_conversation_mock(chunks=("Hello", " world"), system_message=None):
    """Return a mock conversation graph that streams the given chunks."""
    state_values = {}
    if system_message:
        state_values["system_message"] = system_message

    async def _stream_events(turn_input, config, version):
        for chunk in chunks:
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "get_ai_response"},
                "data": {"chunk": AIMessageChunk(content=chunk)},
            }

    conv = MagicMock()
    conv.astream_events = _stream_events
    conv.aget_state = AsyncMock(return_value=MagicMock(values=state_values))
    return conv


def _make_eos_mock():
    eos = MagicMock()
    eos.ainvoke = AsyncMock(return_value={})
    return eos


@pytest.fixture()
def wired_app():
    """Populate app.state with mocks so endpoints work without a real lifespan."""
    app.state.conversation = _make_conversation_mock()
    app.state.eos = _make_eos_mock()
    app.state.session_store = MagicMock()
    app.state.user_profile = MagicMock()
    app.state.seed_context = []
    app.state.new_sessions = set()
    return app


@pytest.fixture()
def client(wired_app):
    return AsyncClient(transport=ASGITransport(app=wired_app), base_url="http://test")


async def test_health_returns_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_create_session_returns_uuid(client):
    r = await client.post("/sessions")
    assert r.status_code == 201
    body = r.json()
    assert "session_id" in body
    assert len(body["session_id"]) == 36  # uuid4 canonical form


async def test_create_session_marks_session_as_new(wired_app, client):
    r = await client.post("/sessions")
    session_id = r.json()["session_id"]
    assert session_id in wired_app.state.new_sessions


async def test_chat_streams_sse_tokens(client):
    r = await client.post("/chat/test-session", json={"message": "hello"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = _parse_sse(r.text)
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) > 0


async def test_chat_stream_ends_with_done(client):
    r = await client.post("/chat/test-session", json={"message": "hello"})
    events = _parse_sse(r.text)
    assert events[-1]["event"] == "done"


async def test_chat_injects_bootstrap_on_first_turn(wired_app, client):
    """First /chat for a new session_id injects user_profile + seed_context."""
    session_id = "new-session-bootstrap"
    wired_app.state.new_sessions.add(session_id)

    # Capture the turn_input the graph receives.
    received = {}

    async def _capturing_stream(turn_input, config, version):
        received.update(turn_input)
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "get_ai_response"},
            "data": {"chunk": AIMessageChunk(content="ok")},
        }

    wired_app.state.conversation.astream_events = _capturing_stream

    await client.post(f"/chat/{session_id}", json={"message": "hi"})

    assert "user_profile" in received, "Bootstrap: user_profile missing from first turn"
    assert "recent_messages" in received, "Bootstrap: recent_messages missing from first turn"
    assert session_id not in wired_app.state.new_sessions, "Bootstrap flag not cleared"


async def test_chat_does_not_inject_bootstrap_on_second_turn(wired_app, client):
    """Subsequent turns for an already-seen session must NOT re-inject bootstrap."""
    session_id = "returning-session"
    # NOT in new_sessions — simulates an already-started session.

    received = {}

    async def _capturing_stream(turn_input, config, version):
        received.update(turn_input)
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "get_ai_response"},
            "data": {"chunk": AIMessageChunk(content="ok")},
        }

    wired_app.state.conversation.astream_events = _capturing_stream

    await client.post(f"/chat/{session_id}", json={"message": "follow-up"})

    assert "user_profile" not in received
    assert "recent_messages" not in received


async def test_chat_quit_returns_system_and_done_without_graph_call(wired_app, client):
    """/quit returns a system SSE event; the conversation graph is NOT invoked."""
    call_count = 0

    async def _tracking_stream(turn_input, config, version):
        nonlocal call_count
        call_count += 1
        yield {}

    wired_app.state.conversation.astream_events = _tracking_stream

    r = await client.post("/chat/test-session", json={"message": "/quit"})
    events = _parse_sse(r.text)

    assert call_count == 0, "Graph should not be invoked on /quit"
    assert any(e["event"] == "system" for e in events)
    assert events[-1]["event"] == "done"


async def test_end_session_invokes_eos_and_returns_202(wired_app, client):
    r = await client.delete("/sessions/some-session")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "saved"
    assert body["session_id"] == "some-session"
    wired_app.state.eos.ainvoke.assert_called_once()


async def test_end_session_clears_new_sessions_flag(wired_app, client):
    """DELETE should remove the session from new_sessions if it was never chatted."""
    wired_app.state.new_sessions.add("orphan-session")
    await client.delete("/sessions/orphan-session")
    assert "orphan-session" not in wired_app.state.new_sessions


async def test_end_session_returns_500_when_eos_fails(wired_app, client):
    wired_app.state.eos.ainvoke = AsyncMock(side_effect=RuntimeError("pipeline failed"))
    r = await client.delete("/sessions/bad-session")
    assert r.status_code == 500


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_sse(body: str) -> list[dict]:
    """Parse an SSE body into a list of {event, data} dicts."""
    results = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        event_type, data_str = None, "{}"
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
        if event_type:
            try:
                results.append({"event": event_type, "data": json.loads(data_str)})
            except json.JSONDecodeError:
                results.append({"event": event_type, "data": {}})
    return results
