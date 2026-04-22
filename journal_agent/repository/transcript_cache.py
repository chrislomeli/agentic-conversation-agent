"""transcript_cache.py — In-memory turn accumulator (buffer only).

``TranscriptStore`` accumulates human/AI turns into Exchange pairs during a
live session and buffers them in memory. Persistence and format conversion
are delegated to an injected ``TranscriptRepository``.

Turn lifecycle:  on_human_turn() → on_ai_turn() → Exchange appended to buffer
Flush:           store_cache(session_id) delegates to repository, then clears buffer
Retrieval:       retrieve_transcript() delegates to repository
"""

from journal_agent.model.session import Role, Turn, Exchange


class TranscriptStore:
    """In-memory turn accumulator. Persistence and format conversion are delegated."""

    def __init__(self, repository=None):
        self._repository = repository
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

    def retrieve_transcript(self, criteria: str | None = None) -> list | None:
        """Load the most recent saved session as LangChain messages, or None."""
        if self._repository is not None:
            return self._repository.retrieve_transcript()
        return None

    def store_cache(self, session_id: str):
        """Flush buffered exchanges to repository and clear the buffer."""
        if self._repository is not None:
            self._repository.save_collection(session_id, self._exchanges)
        self._exchanges = []






