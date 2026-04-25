"""SSE streaming helpers for the FastAPI chat endpoint.

An SSE event on the wire:

    event: token
    data: {"text": "Hello "}

    event: done
    data: {"text": ""}

Each field ends with \\n; events are separated by a blank line (\\n\\n).
"""
import json
from collections.abc import AsyncGenerator, AsyncIterator

from langgraph.graph.state import CompiledStateGraph

from journal_agent.api.models import SseEvent

_AI_RESPONSE_NODE = "get_ai_response"


def format_sse(event: SseEvent, data: str) -> str:
    """Format one SSE event as a wire string."""
    payload = json.dumps({"text": data})
    return f"event: {event}\ndata: {payload}\n\n"


async def graph_stream(
    events: AsyncIterator,
    conversation: CompiledStateGraph,
    config: dict,
) -> AsyncGenerator[str, None]:
    """Consume ``astream_events(version="v2")`` and yield SSE strings.

    Token events:
        Filters ``on_chat_model_stream`` events from the ``get_ai_response``
        node and yields one ``token`` SSE event per non-empty chunk.

    System message:
        After the stream ends, reads the graph's checkpointed state.  If the
        graph left a ``system_message`` (e.g. from ``/save`` confirmations in
        the CAPTURE node), it is emitted as a ``system`` SSE event.

    Terminal events:
        Always ends with a ``done`` event.  On exception, emits ``error``
        instead and does not re-raise (the StreamingResponse body is already
        open; raising would just close the connection silently).
    """
    try:
        async for event in events:
            if event["event"] == "on_chat_model_stream":
                if event.get("metadata", {}).get("langgraph_node") == _AI_RESPONSE_NODE:
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield format_sse(SseEvent.TOKEN, chunk.content)

        # Surface any system_message the graph produced during this turn.
        snapshot = await conversation.aget_state(config)
        if msg := snapshot.values.get("system_message"):
            yield format_sse(SseEvent.SYSTEM, msg)

        yield format_sse(SseEvent.DONE, "")

    except Exception as exc:
        yield format_sse(SseEvent.ERROR, str(exc))
