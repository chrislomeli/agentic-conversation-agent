import uuid
from dataclasses import dataclass

from journal_agent.model.session import Role, Turn, Exchange
from journal_agent.storage.storage import SessionDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def to_messages(exchanges: list[Exchange]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []

    def insert_turn(turn: Turn):
        if turn.role == Role.HUMAN:
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == Role.AI:
            messages.append(AIMessage(content=turn.content))
        elif turn.role == Role.SYSTEM:
            messages.append(SystemMessage(content=turn.content))

    for exchange in exchanges:
        insert_turn(exchange.human)
        insert_turn(exchange.ai)

    return messages


# we are doing two things here - retrieving context and caching turns - but this is just to keep it out of core logic and will be refactored
class SessionStore:
    def __init__(self):
        self._session_store = SessionDatabase()
        self._exchanges: list[Exchange] = []
        self._current_exchange: Exchange = Exchange()

    def on_human_turn(self, session_id: str, role: Role, content: str):
        self._current_exchange.human = Turn(session_id=session_id, role=role, content=content)

    def on_ai_turn(self, session_id: str, role: Role, content: str):
        self._current_exchange.session_id = session_id
        self._current_exchange.ai = Turn(session_id=session_id, role=role, content=content)
        self.cache_exchange()

    def cache_exchange(self):
        self._exchanges.append(self._current_exchange)
        self._current_exchange = Exchange()

    def get_cached_exchanges(self) -> list[Exchange]:
        return self._exchanges

    def retrieve_context(self, criteria: str | None = None) -> list[BaseMessage] | None:
        _messages: list[BaseMessage] | None = None

        if (latest_session_id := self._session_store.get_last_session_id()) is not None:
            if (retrieved_exchanges := self._session_store.load_session(latest_session_id)) is not None:
                _messages = to_messages(retrieved_exchanges)
        return _messages

    def store_cache(self, session_id: str):
        self._session_store.save_session(session_id, self._exchanges)
        self._exchanges = []


if __name__ == "__main__":
    c = SessionStore()
    _session_id = str(uuid.uuid4())
    c.on_human_turn(session_id=_session_id, role=Role.HUMAN, content="test1")
    c.on_ai_turn(session_id=_session_id, role=Role.AI, content="test1")

    c.on_human_turn(session_id=_session_id, role=Role.HUMAN, content="test2")
    c.on_ai_turn(session_id=_session_id, role=Role.AI, content="test2")

    c.on_human_turn(session_id=_session_id, role=Role.HUMAN, content="What is a good project name for a python application that prints hello world?")
    c.on_ai_turn(session_id=_session_id, role=Role.AI, content="hello_world")

    c.store_cache(session_id=_session_id)
    messages = c.retrieve_context()
    print("done")
    print(c.get_cached_exchanges())
