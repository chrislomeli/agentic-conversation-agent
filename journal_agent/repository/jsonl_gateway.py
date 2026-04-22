"""jsonl_gateway.py — JSON-lines access layer for session artifacts.

Provides JsonlGateway, a simple append-only access layer that writes Pydantic
models as one-JSON-object-per-line (.jsonl) files under
``<project-root>/data/<folder>/``.

Each pipeline artifact (transcripts, threads, classified_threads, fragments)
gets its own JsonlGateway instance with a different folder name.

The project root is resolved via the JOURNAL_AGENT_ROOT env var or by
walking up from this file to find pyproject.toml.
"""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from journal_agent.model.session import Exchange
from journal_agent.repository.utils import resolve_project_root

T = TypeVar("T", bound=BaseModel)


class JsonlGateway:
    """Append-only JSONL access layer for Pydantic models.

    Data lives at ``<project-root>/data/<folder>/<session_id>.jsonl``.
    Each call to ``save_json`` appends; ``load_session`` reads all lines back.
    """

    def __init__(self, folder: str = "sessions"):
        self._path = resolve_project_root() / "data" / folder
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

    def get_last_session_id(self) -> str | None:
        """Return the session_id of the most recently created file, or None."""
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

    def save_json(self, session_id: str, exchanges: list[BaseModel]):
        if not exchanges:
            return

        file = self._path / f"{session_id}.jsonl"

        # Append line by line
        with file.open(mode="a", encoding="utf-8") as f:
            for t in exchanges:
                f.write(f"{t.model_dump_json()}\n")  # Manually add newline characters

    def load_session(self, session_id: str, model: type[T] = Exchange) -> list[T] | None:
        """Read all records from the session's JSONL file, or None if empty/missing.

        *model* controls deserialization — defaults to Exchange for backward
        compatibility, but callers can pass ThreadSegment, Fragment, etc.
        """
        file = self._path / f"{session_id}.jsonl"
        data = []
        if file.exists():
            with file.open("r") as f:
                for line in f:
                    t = model.model_validate(json.loads(line.strip()))
                    data.append(t)

        return data if len(data) > 0 else None
