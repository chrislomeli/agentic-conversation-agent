import dataclasses

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from model import Turn, Role


@dataclasses.dataclass
class SessionStore:
    _storage: dict[str, list[Turn]] = dataclasses.field(default_factory=dict)

    def save_session(self, session_id: str, turn: list[Turn]):
        self._storage[session_id] = turn

    def _load_session(self, session_id: str) -> list[Turn] | None:
        return self._storage.get(session_id) or None

    def load_session_messages(self, session_id: str) ->  list[BaseMessage]:
        session_data = self._load_session(session_id)
        messages: list[BaseMessage] = []

        if not session_data:
            return messages

        for value in session_data:
            if value.role == Role.HUMAN:
                messages.append(HumanMessage(content=value.content))
            elif value.role == Role.AI:
                messages.append(AIMessage(content=value.content))

        return messages

    def load_session_turns(self, session_id: str) ->  list[Turn]:
        return self._storage.get(session_id) or []
