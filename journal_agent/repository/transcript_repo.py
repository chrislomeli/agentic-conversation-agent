"""transcript_repo.py — Write-through repository for Exchange records."""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from journal_agent.model.session import Exchange
from journal_agent.repository.jsonl_gateway import JsonlGateway
from journal_agent.repository.pg_gateway import PgGateway
from journal_agent.repository.utils import exchanges_to_messages

T = TypeVar("T", bound=BaseModel)


class TranscriptRepository:
    """Exchange records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway

    def save_collection(self, name: str, items: list[Exchange]) -> None:
        self._jsonl.save_json(name, items)
        self._pg.upsert_exchanges(name, items)

    def load_collection(self, name: str, model: type[T] = Exchange) -> list[T] | None:
        rows = self._pg.fetch_exchanges(name)
        return rows or None

    def get_last_session_id(self) -> str | None:
        return self._pg.get_last_session_id()

    def retrieve_transcript(self) -> list[BaseMessage] | None:
        """Load the most recent saved session as LangChain messages, or None."""
        latest = self.get_last_session_id()
        if latest is None:
            return None
        exchanges = self.load_collection(latest)
        if exchanges is None:
            return None
        return exchanges_to_messages(exchanges)
