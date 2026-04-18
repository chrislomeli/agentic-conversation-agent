"""exchange_store.py — Transcript accumulation and message conversion.

Provides two things:

1. ``to_messages()`` — converts a list of Exchange records into LangChain
   BaseMessage objects, suitable for passing to an LLM.

2. ``TranscriptStore`` — accumulates human/AI turns into Exchange pairs
   during a live session, buffers them in memory, and delegates persistence
   to a JsonStore.

   Turn lifecycle:  on_human_turn() → on_ai_turn() → Exchange appended to buffer
   Flush:           store_cache(session_id) writes buffer to disk and clears it
   Retrieval:       retrieve_transcript() loads the most recent saved session
"""

from journal_agent.model.session import Role, Turn, Exchange
from journal_agent.storage.storage import JsonStore
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def to_messages(exchanges: list[Exchange]) -> list[BaseMessage]:
    """Convert Exchange records to a flat list of LangChain messages.

    Ordering: for each exchange, human turn first, then AI turn.
    Useful for seeding a new session with context from a prior one.
    """
    messages: list[BaseMessage] = []

    def insert_turn(turn: Turn):
        if turn.role == Role.HUMAN:
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == Role.AI:
            messages.append(AIMessage(content=turn.content))
        elif turn.role == Role.SYSTEM:
            messages.append(SystemMessage(content=turn.content))

    for exchange in exchanges:
        if exchange.human:
            insert_turn(exchange.human)
        if exchange.ai:
            insert_turn(exchange.ai)

    return messages


class TranscriptStore:
    """In-memory turn accumulator backed by JsonStore for persistence.

    Nodes call ``on_human_turn`` and ``on_ai_turn`` during the conversation
    loop. Each completed pair becomes an Exchange, buffered in ``_exchanges``.
    At session end, ``store_cache`` flushes the buffer to disk.
    """
    def __init__(self):
        self._session_store = JsonStore("transcripts")
        self._exchanges: list[Exchange] = []
        self._current_exchange: Exchange = Exchange()

    def on_human_turn(self, session_id: str, role: Role, content: str):
        """Record the human half of the current exchange."""
        self._current_exchange.human = Turn(session_id=session_id, role=role, content=content)

    def on_ai_turn(self, session_id: str, role: Role, content: str) -> Exchange:
        """Record the AI half, finalize the exchange, and buffer it."""
        self._current_exchange.session_id = session_id
        self._current_exchange.ai = Turn(session_id=session_id, role=role, content=content)
        response = self._current_exchange
        response.timestamp = response.ai.timestamp
        self._exchanges.append(response)
        self._current_exchange = Exchange()
        return response

    def retrieve_transcript(self, criteria: str | None = None) -> list[BaseMessage] | None:
        """Load the most recent saved session as LangChain messages, or None."""
        _messages: list[BaseMessage] | None = None

        if (latest_session_id := self._session_store.get_last_session_id()) is not None:
            if (retrieved_exchanges := self._session_store.load_session(latest_session_id)) is not None:
                _messages = to_messages(retrieved_exchanges)
        return _messages

    def store_cache(self, session_id: str):
        """Flush buffered exchanges to disk and clear the buffer."""
        self._session_store.save_session(session_id, self._exchanges)
        self._exchanges = []






