import uuid

from journal_agent.model.session import Role, Turn
from journal_agent.storage.storage import SessionDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def to_messages(turns: list[Turn]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for value in turns:
        if value.role == Role.HUMAN:
            messages.append(HumanMessage(content=value.content))
        elif value.role == Role.AI:
            messages.append(AIMessage(content=value.content))
        elif value.role == Role.SYSTEM:
            messages.append(SystemMessage(content=value.content))
    return messages


class Exchange:
    def __init__(self):
        self.exchange_id: str = str(uuid.uuid4())
        self.human: Turn | None = None
        self.ai = Turn | None



# we are doing two things here - retrieving context and caching turns - but this is just to keep it out of core logic and will be refactored
class SessionStore:
    def __init__(self):
        self._session_store = SessionDatabase()
        self._exchanges: list[Exchange] = []
        self._current_exchange: Exchange = Exchange()

    def on_human_turn(self, session_id: str, role: Role, content: str, metadata: dict | None = None):
        self._current_exchange.human = Turn(session_id=session_id, role=role, content=content, metadata=metadata)

    def on_ai_turn(self, session_id: str, role: Role, content: str, metadata: dict | None = None):
        self._current_exchange.ai = Turn(session_id=session_id, role=role, content=content, metadata=metadata)
        self.cache_exchange()

    def cache_exchange(self):
        self._exchanges.append(self._current_exchange)
        self._current_exchange = Exchange()

    def cache_turn(self, session_id: str, role: Role, content: str, metadata: dict | None = None):
        self._turns.append(
            Turn(session_id=session_id, role=role, content=content, metadata=metadata)
        )

    def get_cached_turns(self) -> list[Turn]:
        return self._turns

    def retrieve_context(self, criteria: str | None = None) -> list[BaseMessage] | None:
        messages: list[BaseMessage] | None = None

        if (latest_session_id := self._session_store.get_last_session_id()) is not None:
            if (retrieved_turns := self._session_store.load_session(latest_session_id)) is not None:
                messages = to_messages(retrieved_turns)
        return messages

    def store_cache(self, session_id: str):
        self._session_store.save_session(session_id, self._turns)
        self._turns = []
