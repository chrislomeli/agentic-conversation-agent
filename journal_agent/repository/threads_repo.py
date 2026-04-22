"""threads_repo.py — Write-through repository for ThreadSegment records.

Reused for both threads/ and classified_threads/ folders — the PG table is merged;
upsert COALESCEs tags so classification overlays the initial write.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from journal_agent.model.session import ThreadSegment
from journal_agent.repository.jsonl_gateway import JsonlGateway
from journal_agent.repository.pg_gateway import PgGateway

T = TypeVar("T", bound=BaseModel)


class ThreadsRepository:
    """ThreadSegment records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway

    def save_collection(self, name: str, items: list[ThreadSegment]) -> None:
        self._jsonl.save_json(name, items)
        for thread in items:
            self._pg.upsert_thread(name, thread)

    def load_collection(self, name: str, model: type[T] = ThreadSegment) -> list[T] | None:
        rows = self._pg.fetch_threads(name)
        return rows or None

    def get_last_session_id(self) -> str | None:
        return self._pg.get_last_session_id()
