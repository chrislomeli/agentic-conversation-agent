"""capture_repo.py — pgvector-backed store for user-initiated captures.

Captures are deliberate notes saved via /save <topic> <text>.
They live in the `captures` table (same schema as `fragments`) but are
intentionally separate so the Phase 11 reflection pipeline never picks
them up for stance classification.
"""

from __future__ import annotations

import logging

from journal_agent.model.session import Fragment
from journal_agent.stores.embedder import Embedder
from journal_agent.stores.pg_gateway import PgGateway, get_pg_gateway

logger = logging.getLogger(__name__)


class CaptureRepository:
    """Capture store backed by Postgres + pgvector."""

    def __init__(self, pg_gateway: PgGateway | None = None, embedder: Embedder | None = None):
        self._pg = pg_gateway or get_pg_gateway()
        self._embedder = embedder or Embedder()

    def search_captures(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Fragment, float]]:
        """Embed the query, then return cosine top-k from the captures table."""
        query_vec = self._embedder.embed(query_text)
        return self._pg.search_captures_similar(query_vec, top_k=top_k, min_score=min_relevance)

    def save_captures(self, fragments: list[Fragment]) -> None:
        """Embed all captures in one batch pass, then upsert to Postgres."""
        if not fragments:
            return
        texts = [
            f"{f.content} {' '.join(t.tag for t in f.tags)}"
            for f in fragments
        ]
        embeddings = self._embedder.embed_batch(texts)
        for fragment, vec in zip(fragments, embeddings):
            self._pg.upsert_capture(fragment, embedding=vec)
