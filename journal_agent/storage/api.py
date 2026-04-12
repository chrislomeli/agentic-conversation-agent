from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from journal_agent.model.storage import Turn, Role
from .storage import SessionDatabase

def to_messages(turns: list[Turn]) -> list[BaseMessage] | None:
    messages: list[BaseMessage] = []
    for value in turns:
        if value.role == Role.HUMAN:
            messages.append(HumanMessage(content=value.content))
        elif value.role == Role.AI:
            messages.append(AIMessage(content=value.content))
        elif value.role == Role.SYSTEM:
            messages.append(SystemMessage(content=value.content))
    return messages

# we are doing two things here - retrieving context and caching turns - but this is just to keep it out of core logic and will be refactored
class SessionStore:
    def __init__(self):
        self._session_store = SessionDatabase()
        self._turns: list[Turn] = []

    def cache_turn(self, session_id: str, role: Role, content: str, metadata: dict | None = None):
        self._turns.append(Turn(session_id=session_id, role=role, content=content, metadata=metadata))

    def retrieve_context(self, criteria: str = None) -> list[BaseMessage] | None:
        messages: list[BaseMessage] | None = None
        turns: list[Turn] = []

        if ( latest_session_id := self._session_store.get_last_session_id() ) is not None:
            if (retrieved_turns:= self._session_store.load_session(latest_session_id)) is not None:
                turns.extend(retrieved_turns)
                messages = to_messages(retrieved_turns)
        return messages

    def store_cache(self, session_id: str):
        self._session_store.save_session(session_id, self._turns)
        self._turns = []
