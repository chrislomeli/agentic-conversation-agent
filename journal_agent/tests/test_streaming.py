"""Tests for streaming get_ai_response (design/api-build-plan.md item #9a).

The node now consumes ``llm.astream(messages)`` and accumulates chunks into
a final AIMessage. Terminal printing has moved to the runner via
``stream_ai_response_to_terminal``.

What the tests prove:
    1. Chunks from the LLM stream accumulate into the final AIMessage.
    2. The node no longer calls ``talk_to_human`` — terminal coupling is gone.
    3. The terminal consumer prints chunks for ``get_ai_response`` only;
       classifier/extractor LLM events are filtered out.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from journal_agent.comms.human_chat import stream_ai_response_to_terminal
from journal_agent.graph import journal_graph as jg
from journal_agent.graph.journal_graph import make_get_ai_response
from journal_agent.graph.state import JournalState
from journal_agent.model.session import ContextSpecification, StatusValue


# ── helpers ──────────────────────────────────────────────────────────────────


async def _achunks(*texts):
    for t in texts:
        yield AIMessageChunk(content=t)


async def _events(*items):
    for it in items:
        yield it


def _stream_event(node: str, content: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": AIMessageChunk(content=content)},
    }


def _end_event(node: str) -> dict:
    return {
        "event": "on_chat_model_end",
        "metadata": {"langgraph_node": node},
        "data": {},
    }


# ── make_get_ai_response: streaming behavior ─────────────────────────────────


async def test_streaming_chunks_accumulate_into_ai_message(monkeypatch):
    """Each chunk yielded by llm.astream contributes to the final content."""
    monkeypatch.setattr(jg, "get_prompt", lambda key, state: "system prompt")

    context_builder = MagicMock()
    context_builder.get_context.return_value = []

    llm = MagicMock()
    llm.astream = lambda messages: _achunks("hello ", "world", "!")

    session_store = MagicMock()
    session_store.on_ai_turn.return_value = MagicMock()

    node = make_get_ai_response(
        llm=llm,
        session_store=session_store,
        context_builder=context_builder,
    )

    state = JournalState(
        session_id="s1", context_specification=ContextSpecification()
    )
    result = await node(state)

    assert isinstance(result["session_messages"][0], AIMessage)
    assert result["session_messages"][0].content == "hello world!"
    assert result["status"] == StatusValue.PROCESSING

    # The exchange was persisted with the full assembled content.
    session_store.on_ai_turn.assert_called_once()
    assert session_store.on_ai_turn.call_args.kwargs["content"] == "hello world!"


async def test_journal_graph_module_does_not_import_talk_to_human(monkeypatch):
    """Static guard: the graph module must not import the terminal printer.

    The node can't accidentally call ``talk_to_human`` if the symbol isn't in
    the module. This catches a regression where someone re-adds the terminal
    coupling we removed during #9a / #9c. If you legitimately need to add
    a printer somewhere, do it in the runner, not the graph.
    """
    monkeypatch.setattr(jg, "get_prompt", lambda key, state: "system prompt")

    assert not hasattr(jg, "talk_to_human"), (
        "journal_graph imports talk_to_human — terminal coupling has crept back in. "
        "Move terminal output to the runner (main.py / API endpoint)."
    )

    # Behavioral sanity: the streaming node runs end-to-end without it.
    context_builder = MagicMock()
    context_builder.get_context.return_value = []

    llm = MagicMock()
    llm.astream = lambda messages: _achunks("ok")

    session_store = MagicMock()
    session_store.on_ai_turn.return_value = MagicMock()

    node = make_get_ai_response(
        llm=llm,
        session_store=session_store,
        context_builder=context_builder,
    )

    state = JournalState(
        session_id="s1", context_specification=ContextSpecification()
    )
    result = await node(state)
    assert result["session_messages"][0].content == "ok"


# ── stream_ai_response_to_terminal: rendering and filtering ──────────────────


async def test_terminal_consumer_prints_ai_chunks(capsys):
    """Chunks from get_ai_response render incrementally with the AI prefix."""
    events = _events(
        _stream_event("get_ai_response", "hello "),
        _stream_event("get_ai_response", "world"),
        _end_event("get_ai_response"),
    )

    await stream_ai_response_to_terminal(events)

    out = capsys.readouterr().out
    assert "AI:" in out
    assert "hello world" in out


async def test_terminal_consumer_filters_non_ai_node_events(capsys):
    """Events from classifier / extractor nodes must not leak to stdout."""
    events = _events(
        _stream_event("intent_classifier", "secret-classifier-output"),
        _stream_event("exchange_decomposer", "secret-decomposer-output"),
        _stream_event("get_ai_response", "visible"),
        _end_event("get_ai_response"),
    )

    await stream_ai_response_to_terminal(events)

    out = capsys.readouterr().out
    assert "secret-classifier-output" not in out
    assert "secret-decomposer-output" not in out
    assert "visible" in out


async def test_terminal_consumer_handles_empty_stream(capsys):
    """An empty event stream should produce no output (no dangling prefix)."""
    events = _events()
    await stream_ai_response_to_terminal(events)
    out = capsys.readouterr().out
    assert out == ""
