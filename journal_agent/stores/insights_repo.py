"""insights_repo.py — Write-through stores for Insight records."""

from __future__ import annotations

from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from journal_agent.model.session import Insight
from journal_agent.stores.jsonl_gateway import JsonlGateway
from journal_agent.stores.pg_gateway import PgGateway

T = TypeVar("T", bound=BaseModel)


class InsightsRepository:
    """Insight records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway

    def save_insights(self, items: list[Insight]) -> None:
        if len(items) > 0:
            file_name = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._jsonl.save_json(file_name, items)
            self._pg.upsert_insights(items)

    def load_insights(self, search_label: str | None = None, date_cutoff: datetime | None = None) -> list[Insight] | None:
        rows = self._pg.fetch_insights(search_label, date_cutoff)
        return rows or None
