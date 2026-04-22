"""fragment_repo.py — pgvector-backed fragment stores.

    save_fragments()    — embeds content + upserts to fragments + junctions
    search_fragments()  — cosine similarity search via pgvector
    load_all()          — full fragment scan (no embedding required)

Embeddings are generated locally via fastembed (all-MiniLM-L6-v2, 384-dim)
before being written to Postgres — so the embedding model lives here, not
inside the database.
"""

from __future__ import annotations

import logging

from journal_agent.model.session import Fragment
from journal_agent.stores.embedder import Embedder
from journal_agent.stores.pg_gateway import PgGateway, get_pg_gateway

logger = logging.getLogger(__name__)


class PgFragmentRepository:
    """Fragment stores backed entirely by Postgres + pgvector."""

    def __init__(self, pg_gateway: PgGateway | None = None, embedder: Embedder | None = None):
        self._pg = pg_gateway or get_pg_gateway()
        self._embedder = embedder or Embedder()

    def save_fragments(self, fragments: list[Fragment]) -> None:
        """Embed all fragments in one batch pass, then upsert to Postgres."""
        if not fragments:
            return
        texts = [f.content for f in fragments]
        embeddings = self._embedder.embed_batch(texts)
        for fragment, vec in zip(fragments, embeddings):
            self._pg.upsert_fragment(fragment, embedding=vec)

    def search_fragments(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Fragment, float]]:
        """Embed the query, then return cosine top-k from pgvector."""
        query_vec = self._embedder.embed(query_text)
        return self._pg.search_similar(query_vec, top_k=top_k, min_score=min_relevance)

    def load_all(self, user_id: str | None = None) -> list[Fragment]:
        """Return all fragments."""
        return self._pg.fetch_fragments()
