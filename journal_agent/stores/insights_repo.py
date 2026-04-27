"""insights_repo.py — Write-through stores for Insight records."""

from __future__ import annotations

from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from journal_agent.graph.state import WindowParams
from journal_agent.model.session import Insight
from journal_agent.stores.embedder import Embedder
from journal_agent.stores.jsonl_gateway import JsonlGateway
from journal_agent.stores.pg_gateway import PgGateway

T = TypeVar("T", bound=BaseModel)


class InsightsRepository:
    """Insight records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(
        self,
        jsonl_gateway: JsonlGateway,
        pg_gateway: PgGateway,
        embedder: Embedder | None = None,
    ):
        self._jsonl = jsonl_gateway
        self._pg = pg_gateway
        self._embedder = embedder or Embedder()

    def save_insights(self, items: list[Insight]) -> None:
        if not items:
            return
        texts = [f"{i.label}\n\n{i.body}" for i in items]
        vectors = self._embedder.embed_batch(texts)
        for insight, vec in zip(items, vectors):
            insight.embedding = vec.tolist()
        file_name = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._jsonl.save_json(file_name, items)
        self._pg.upsert_insights(items)

    def load_insights(self, fetch_params: WindowParams | None = None ) -> list[Insight]:
        return self._pg.fetch_insights(fetch_params) or []
