"""protocols.py — Storage protocols for the journal agent.

Defines the contracts that all storage implementations must satisfy.
The rest of the application depends on these protocols, never on concrete
store classes.  This allows swapping JSONL/Chroma for Postgres/pgvector
(or any other backend) with zero changes outside main.py.

Concrete implementations:
    Local (no infrastructure):
        JsonArtifactStore   — JSONL files  (storage.py)
        TranscriptStore     — in-memory + JSONL  (exchange_store.py)
        VectorStore         — ChromaDB  (vector_store.py)
        UserProfileStore    — single JSON file  (profile_store.py)
    Production:
        PostgresArtifactStore, PgVectorStore, PgProfileStore, etc.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from journal_agent.model.session import Exchange, Fragment, Role, UserProfile

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class ArtifactStore(Protocol):
    """Generic session-keyed persistence for pipeline intermediates
    (transcripts, threads, classified_threads, fragments-as-jsonl)."""

    def save_session(self, session_id: str, items: list[BaseModel]) -> None: ...
    def load_session(self, session_id: str, model: type[T] = ...) -> list[T] | None: ...
    def get_last_session_id(self) -> str | None: ...


@runtime_checkable
class FragmentStore(Protocol):
    """Unified fragment persistence + vector search.

    Replaces the split between JsonStore("fragments") and VectorStore.
    A single call to ``save_fragments`` handles both structured persistence
    and embedding indexing.
    """

    def save_fragments(self, fragments: list[Fragment]) -> None: ...

    def search_fragments(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Fragment, float]]: ...

    def load_all(self, user_id: str | None = None) -> list[Fragment]: ...


@runtime_checkable
class ProfileStore(Protocol):
    """User profile persistence."""

    def load_profile(self, user_id: str | None = None) -> UserProfile | None: ...
    def save_profile(self, profile: UserProfile, user_id: str | None = None) -> None: ...


@runtime_checkable
class SessionStore(Protocol):
    """Exchange accumulation during a live session + transcript retrieval.

    In-memory buffering of human/AI turn pairs, flushed to durable storage
    at session end via ``store_cache``.
    """

    def on_human_turn(self, session_id: str, role: Role, content: str) -> None: ...
    def on_ai_turn(self, session_id: str, role: Role, content: str) -> Exchange: ...
    def retrieve_transcript(self, criteria: str | None = None) -> list[BaseMessage] | None: ...
    def store_cache(self, session_id: str) -> None: ...
