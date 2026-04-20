"""chroma_fragment_store.py — Local FragmentStore backed by ChromaDB + JSONL.

Satisfies the ``FragmentStore`` protocol by combining:
    - VectorStore (ChromaDB) for embedding-based search
    - JsonStore ("fragments") for structured JSONL persistence

A single call to ``save_fragments`` writes to both stores atomically
(from the caller's perspective).  This eliminates the two-node pipeline
(save_fragments_to_json → save_fragments_to_vectordb) in the graph.
"""

import json
import logging
from pathlib import Path

from journal_agent.model.session import Fragment
from journal_agent.storage.storage import JsonStore
from journal_agent.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChromaFragmentStore:
    """Local FragmentStore: JSONL for persistence, ChromaDB for vector search."""

    def __init__(
        self,
        json_store: JsonStore | None = None,
        vector_store: VectorStore | None = None,
    ):
        self._json_store = json_store or JsonStore("fragments")
        self._vector_store = vector_store or VectorStore()

    def save_fragments(self, fragments: list[Fragment]) -> None:
        if not fragments:
            return

        session_id = fragments[0].session_id
        self._json_store.save_session(session_id, fragments)
        self._vector_store.add_to_chroma_from_fragments(fragments)

    def search_fragments(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Fragment, float]]:
        return self._vector_store.search_fragments(
            query_text=query_text,
            min_relevance=min_relevance,
            top_k=top_k,
        )

    def load_all(self, user_id: str | None = None) -> list[Fragment]:
        """Load all fragments across all sessions from the JSONL store.

        Globs all session files and concatenates.  Deduplicates by fragment_id
        in case of overlapping writes.
        """
        seen: set[str] = set()
        all_fragments: list[Fragment] = []

        if not self._json_store._path.exists():
            return []

        for jsonl_file in sorted(self._json_store._path.glob("*.jsonl")):
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fragment = Fragment.model_validate_json(line)
                    if fragment.fragment_id not in seen:
                        seen.add(fragment.fragment_id)
                        all_fragments.append(fragment)

        return all_fragments
