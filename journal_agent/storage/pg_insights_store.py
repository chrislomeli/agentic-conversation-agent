"""pg_insight_store.py — pgvector-backed insightStore.

Fully replaces ChromainsightStore when Postgres is enabled.
Satisfies the insightStore protocol:

    save_insights()    — embeds content + upserts to insights + junctions
    search_insights()  — cosine similarity search via pgvector
    load_all()          — full insight scan (no embedding required)

Embeddings are generated locally via fastembed (all-MiniLM-L6-v2, 384-dim)
before being written to Postgres — so the embedding model lives here, not
inside the database.
"""

from __future__ import annotations

import logging

from journal_agent.model.session import insight, Insight
from journal_agent.storage.embedder import Embedder
from journal_agent.storage.pg_store import PgStore, get_pg_store

logger = logging.getLogger(__name__)


class PgInsightStore:
    """insightStore backed entirely by Postgres + pgvector.

    Injected into the graph the same way as ChromainsightStore.
    """

    def __init__(self, pg_store: PgStore | None = None, embedder: Embedder | None = None):
        self._pg = pg_store or get_pg_store()
        self._embedder = embedder or Embedder()

    # ── insightStore protocol ─────────────────────────────────────────────────

    def save_insights(self, insights: list[Insight]) -> None:
        """Embed all insights in one batch pass, then upsert to Postgres."""
        if not insights:
            return
        texts = [f.content for f in insights]
        embeddings = self._embedder.embed_batch(texts)
        for insight, vec in zip(insights, embeddings):
            self._pg.upsert_insight(insight, embedding=vec)

    def search_insights(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Insight, float]]:
        """Embed the query, then return cosine top-k from pgvector."""
        query_vec = self._embedder.embed(query_text)
        return self._pg.search_similar(query_vec, top_k=top_k, min_score=min_relevance)

    def load_all(self, user_id: str | None = None) -> list[insight]:
        """Return all insights, optionally filtered by session via user_id.

        Note: user_id is accepted for protocol compatibility but ignored here —
        all stored insights are returned. Pass session_id directly to
        pg_store.fetch_insights(session_id) if you need per-session filtering.
        """
        return self._pg.fetch_insights()
