"""commands.py — Parse raw user input into structured turn input.

Both the terminal runner and the FastAPI endpoint take a string from the
user and need to produce one of:

    - "quit": exit the session loop and run the end-of-session pipeline
    - a structured ``ParsedInput`` to invoke the conversation graph for one
      turn: ``user_command`` + ``user_command_args`` + the message text that
      becomes the new HumanMessage on session_messages

The parsing logic used to live inside the ``get_user_input`` graph node.
After #9c it lives here so the conversation graph never sees raw input —
the runner produces a typed turn payload, the graph processes it.
"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from journal_agent.model.session import UserCommandValue


@dataclass(frozen=True)
class ParsedInput:
    """Structured turn input handed to the conversation graph.

    Attributes:
        quit: True if the user typed ``/quit``. The runner should exit the
            session loop and trigger the end-of-session pipeline.
        command: Which user command (if any) this turn carries.
        command_args: Raw args after the command word (whitespace-stripped).
        message: The text to land on session_messages as a HumanMessage.
            Empty string means "no message for this turn" (e.g. a ``/save``
            that captures inline text without a conversation prompt).
    """

    quit: bool = False
    command: UserCommandValue = UserCommandValue.NONE
    command_args: str = ""
    message: str = ""


_REFLECT_PROMPT = (
    "Please share the patterns and insights you've observed from my recent "
    "journal entries."
)


def _recall_prompt(args: str) -> str:
    if args:
        return f"Please recall what I've previously written about: {args}"
    return "Please recall my recent journal entries."


def parse_user_input(text: str) -> ParsedInput:
    """Convert a raw user message into a ParsedInput for one turn.

    Recognized commands:
        ``/quit``                 — end the session
        ``/reflect``              — invoke the reflection graph
        ``/recall [topic]``       — search journal history
        ``/save [n] <topic>``     — save the last n exchanges as a fragment
        ``/save <topic> <text>``  — save inline text as a fragment

    Plain text becomes a normal conversation turn (command=NONE, message=text).
    """
    stripped = text.strip()

    if stripped == "/quit":
        return ParsedInput(quit=True)

    if stripped == "/reflect":
        return ParsedInput(
            command=UserCommandValue.REFLECT,
            message=_REFLECT_PROMPT,
        )

    if stripped.startswith("/recall"):
        parts = stripped.split(maxsplit=1)
        args = parts[1].strip() if len(parts) > 1 else ""
        return ParsedInput(
            command=UserCommandValue.RECALL,
            command_args=args,
            message=_recall_prompt(args),
        )

    if stripped.startswith("/save"):
        parts = stripped.split(maxsplit=1)
        args = parts[1].strip() if len(parts) > 1 else ""
        return ParsedInput(
            command=UserCommandValue.SAVE,
            command_args=args,
            message="",
        )

    return ParsedInput(message=stripped)


def build_turn_input(parsed: ParsedInput, *, session_id: str) -> dict:
    """Build the per-turn input dict for the conversation graph.

    Always includes ``session_id`` because it is required on ``JournalState``
    and the first invocation for a fresh thread_id has no prior state to
    inherit it from.

    Runners (terminal, API) can extend the returned dict with bootstrap
    fields like ``user_profile`` or ``recent_messages`` on the first turn
    for a session — those are session-bootstrap concerns, not per-turn.
    """
    turn_input: dict = {
        "session_id": session_id,
        "user_command": parsed.command,
        "user_command_args": parsed.command_args,
        "system_message": None,
    }
    if parsed.message:
        turn_input["session_messages"] = [HumanMessage(content=parsed.message)]
    return turn_input
