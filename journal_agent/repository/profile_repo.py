"""profile_repo.py — Write-through repository for UserProfile."""

from __future__ import annotations

from journal_agent.model.session import UserProfile
from journal_agent.repository.jsonl_gateway import JsonlGateway
from journal_agent.repository.pg_gateway import PgGateway


class UserProfileRepository:
    """UserProfile: local JSON + Postgres user_profiles row."""

    def __init__(self, jsonl_gateway: JsonlGateway, pg_gateway: PgGateway):
        self._json = jsonl_gateway
        self._pg = pg_gateway

    def load_profile(self, user_id: str | None = "default_user") -> UserProfile | None:
        user_profile = self._pg.fetch_profile(user_id)
        user_profile.is_current = True
        user_profile.is_updated = False
        return user_profile

    def save_profile(self, profile: UserProfile) -> None:
        self._json.save_json(profile.user_id, [profile])
        self._pg.upsert_profile(profile)
