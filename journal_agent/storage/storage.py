import dataclasses
import json
from abc import ABC, abstractmethod
from pathlib import Path

from journal_agent.model.turn import Turn
from langchain_core.messages import BaseMessage


@dataclasses.dataclass
class SessionData:
    messages: list[BaseMessage]
    turns: list[Turn]


class DataStore(ABC):
    @abstractmethod
    def save_session(self, session_id: str, turn: list[Turn]):
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> list[Turn] | None:
        pass


class SessionDatabase(DataStore):
    def __init__(self):
        path_name = "data/sessions"  # this is just placeholder internal access - no one else needs to know or set this
        self._path = Path(path_name)
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

    def get_last_session_id(self) -> str | None:
        if not self._path.exists():
            return None

        # Get all files and find the one with the maximum st_ctime
        files = self._path.glob("*")

        # Filter for files only and find the max by creation time (st_ctime)
        try:
            latest_file = max((f for f in files if f.is_file()), key=lambda x: x.stat().st_ctime)
            return latest_file.name.split(".")[0]
        except ValueError:
            return None

    def save_session(self, session_id: str, turn: list[Turn]):
        if self._path is None:
            raise ValueError("Path name is not set")

        if not turn:
            return

        file = self._path / f"{session_id}.jsonl"

        # Append line by line
        with file.open(mode="a", encoding="utf-8") as f:
            for t in turn:
                f.write(f"{t.model_dump_json()}\n")  # Manually add newline characters

    def load_session(self, session_id: str) -> list[Turn] | None:
        file = self._path / f"{session_id}.jsonl"
        data = []
        if file.exists():
            with file.open("r") as f:
                for line in f:
                    t = Turn.model_validate(json.loads(line.strip()))
                    data.append(t)

        return data if len(data) > 0 else None
