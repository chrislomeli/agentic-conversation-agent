from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any


class Speaker(StrEnum):
    AI = "ai"
    SYSTEM = "system"


_ANSI = {
    Speaker.AI:     "\033[36m",   # cyan
    Speaker.SYSTEM: "\033[33m",   # yellow
}
_RESET = "\033[0m"
_PREFIX = {
    Speaker.AI:     "AI",
    Speaker.SYSTEM: "System",
}

# Only stream tokens from this graph node to the terminal. Other LLM calls
# (intent classifier, exchange decomposer, fragment extractor) emit the same
# event type but should not leak to the user-facing console.
_AI_RESPONSE_NODE = "get_ai_response"


def get_console_input() -> str:
    while True:
        print("You (blank line to send):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if text:
            return text


def display_console_output(message: str, speaker: Speaker = Speaker.SYSTEM) -> None:
    color = _ANSI[speaker]
    prefix = _PREFIX[speaker]
    print(f"{color}{prefix}: {message}{_RESET}")


async def stream_ai_response_to_terminal(
    events: AsyncIterator[dict[str, Any]],
) -> None:
    """Consume LangGraph ``astream_events(v2)`` and render AI tokens to stdout.

    Filters by ``langgraph_node == "get_ai_response"`` so classifier and
    extractor LLM calls don't leak into the user-facing stream. On the first
    chunk of a response we print the AI prefix; on the model's end event we
    drop the ANSI reset and a newline. The ``finally`` guards the case where
    the stream is cancelled or errors mid-response so the terminal isn't
    left with dangling color state.
    """

    color = _ANSI[Speaker.AI]
    prefix = _PREFIX[Speaker.AI]
    in_response = False

    try:
        async for event in events:
            ev = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node")
            if node != _AI_RESPONSE_NODE:
                continue

            if ev == "on_chat_model_stream":
                if not in_response:
                    print(f"{color}{prefix}: ", end="", flush=True)
                    in_response = True
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    print(content, end="", flush=True)
            elif ev == "on_chat_model_end" and in_response:
                print(_RESET)
                in_response = False
    finally:
        if in_response:
            print(_RESET)
