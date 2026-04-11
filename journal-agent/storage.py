import dataclasses
import json
from abc import ABC, abstractmethod
from pathlib import Path
import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from model import Turn, Role

@dataclasses.dataclass
class SessionData:
    messages: list[BaseMessage]
    turns: list[Turn]

class DataStore(ABC):
    @abstractmethod
    def save_session(self, session_id: str, turn: list[Turn]):
        pass

    @abstractmethod
    def _load_session(self, session_id: str) -> list[Turn] | None:
        pass

    @abstractmethod
    def load_session_messages(self, session_id: str) -> SessionData | None:
        pass

@dataclasses.dataclass
class SessionStore(DataStore):

    _path: Path

    def __init__(self, path_name: str):
        self._path = Path(path_name)
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)


    def get_last_session_id(self) -> str | None:
        if not self._path.exists():
            return None

        # Get all files and find the one with the maximum st_ctime
        files = self._path.glob('*')

        # Filter for files only and find the max by creation time (st_ctime)
        try:
            latest_file = max((f for f in files if f.is_file()), key=lambda x: x.stat().st_ctime)
            return latest_file.name.split('.')[0]
        except ValueError:
            return None


    def save_session(self, session_id: str, turn: list[Turn]):
        if self._path is None:
            raise ValueError("Path name is not set")

        file = self._path / f"{session_id}.jsonl"

        # Write line by line
        with file.open(mode="w", encoding="utf-8") as file:
            for t in turn:
                file.write(f"{t.model_dump_json()}\n")  # Manually add newline characters

    def _load_session(self, session_id: str) -> list[Turn] | None:
        file = self._path / f"{session_id}.jsonl"
        data = []
        if file.exists():
            with file.open('r') as f:
                for line in f:
                    t = Turn.model_validate(json.loads(line.strip()))
                    data.append(t)

        return data if len(data) > 0 else None

    def load_session_messages(self, session_id: str ) ->  SessionData | None:
        turns = self._load_session(session_id)
        if turns is None:
            return None

        messages: list[BaseMessage]  = []

        for value in turns:
            if value.role == Role.HUMAN:
                messages.append(HumanMessage(content=value.content))
            elif value.role == Role.AI:
                messages.append(AIMessage(content=value.content))
            elif value.role == Role.SYSTEM:
                messages.append(SystemMessage(content=value.content))

        return SessionData(turns=turns, messages=messages)

