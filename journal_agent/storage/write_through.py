"""write_through.py — Store-level write-through wrappers for non-fragment stores.

Fragment storage is handled by PgFragmentStore (pg_fragment_store.py) when
Postgres is enabled — no write-through needed there.

Remaining wrappers:
    WriteThroughTranscriptStore  satisfies ArtifactStore  — Exchange records
    WriteThroughThreadStore      satisfies ArtifactStore  — ThreadSegment records
                                                            (reused for threads/ and
                                                             classified_threads/ folders;
                                                             PG table is merged)
    WriteThroughProfileStore     satisfies ProfileStore   — UserProfile

Read paths delegate to the local store (JSONL stays authoritative during transition).
Enable via settings.enable_postgres in main.py.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from journal_agent.model.session import Exchange, ThreadSegment, UserProfile
from journal_agent.storage.pg_store import PgStore
from journal_agent.storage.profile_store import UserProfileStore
from journal_agent.storage.storage import JsonStore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class WriteThroughTranscriptStore:
    """ArtifactStore for Exchange records: writes to JSONL + Postgres; reads from Postgres."""

    def __init__(self, json_store: JsonStore, pg_store: PgStore):
        self._json = json_store
        self._pg = pg_store

    def save_session(self, session_id: str, items: list[Exchange]) -> None:
        self._json.save_session(session_id, items)
        self._pg.upsert_exchanges(session_id, items)

    def load_session(self, session_id: str, model: type[T] = Exchange) -> list[T] | None:
        rows = self._pg.fetch_exchanges(session_id)
        return rows or None  # protocol: return None on miss, not []

    def get_last_session_id(self) -> str | None:
        return self._pg.get_last_session_id()


class WriteThroughThreadStore:
    """ArtifactStore for ThreadSegment records: writes to JSONL + Postgres; reads from Postgres.

    Reused for both threads/ and classified_threads/ folders — the PG table is merged;
    upsert COALESCEs tags so classification overlays the initial write.
    """

    def __init__(self, json_store: JsonStore, pg_store: PgStore):
        self._json = json_store
        self._pg = pg_store

    def save_session(self, session_id: str, items: list[ThreadSegment]) -> None:
        self._json.save_session(session_id, items)
        for thread in items:
            self._pg.upsert_thread(session_id, thread)

    def load_session(self, session_id: str, model: type[T] = ThreadSegment) -> list[T] | None:
        rows = self._pg.fetch_threads(session_id)
        return rows or None  # protocol: return None on miss, not []

    def get_last_session_id(self) -> str | None:
        return self._pg.get_last_session_id()


class WriteThroughProfileStore:
    """ProfileStore: local JSON + Postgres user_profiles row."""

    def __init__(self, local_store: UserProfileStore, pg_store: PgStore):
        self._local = local_store
        self._pg = pg_store

    def load_profile(self, user_id: str | None = "default_user") -> UserProfile | None:
        # return self._local.load_profile(user_id)
        pg = self._pg.fetch_profile(user_id)
        return pg[0] if pg else None



    def save_profile(self, profile: UserProfile) -> None:
        self._local.save_profile(profile)
        self._pg.upsert_profile(profile)
