"""pg_fragment_store.py — pgvector-backed FragmentStore.

Fully replaces ChromaFragmentStore when Postgres is enabled.
Satisfies the FragmentStore protocol:

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
from journal_agent.storage.embedder import Embedder
from journal_agent.storage.pg_store import PgStore, get_pg_store

logger = logging.getLogger(__name__)


class PgFragmentStore:
    """FragmentStore backed entirely by Postgres + pgvector.

    Injected into the graph the same way as ChromaFragmentStore.
    """

    def __init__(self, pg_store: PgStore | None = None, embedder: Embedder | None = None):
        self._pg = pg_store or get_pg_store()
        self._embedder = embedder or Embedder()

    # ── FragmentStore protocol ─────────────────────────────────────────────────

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
        """Return all fragments, optionally filtered by session via user_id.

        Note: user_id is accepted for protocol compatibility but ignored here —
        all stored fragments are returned. Pass session_id directly to
        pg_store.fetch_fragments(session_id) if you need per-session filtering.
        """
        return self._pg.fetch_fragments()
