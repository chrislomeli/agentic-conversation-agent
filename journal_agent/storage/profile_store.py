from pathlib import Path

from journal_agent.model.session import UserProfile
from journal_agent.storage.utils import resolve_project_root


class UserProfileStore:
    """Local ProfileStore: single JSON file per user.

    Satisfies the ``ProfileStore`` protocol.  The ``user_id`` parameter
    is accepted for forward compatibility but currently ignored (single-file
    store at ``data/profile/profile.json``).
    """
    _path: Path
    def __init__(self):
        self._path = resolve_project_root() / "data" / "profile" / "profile.json"
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

    def load_profile(self, user_id: str | None = None) -> UserProfile | None:
        try:
            if self._path is None:
                raise ValueError("Path name is not set")

            # Write single atomic profile
            with self._path.open(mode="r", encoding="utf-8") as f:
                d = f.read()
                user_profile =  UserProfile.model_validate_json(d)
                user_profile.is_current = True
                user_profile.is_updated = False
                return user_profile

        except Exception as e:
            pass


    def save_profile(self, profile: UserProfile, user_id: str | None = None) -> None:
        if self._path is None:
            raise ValueError("Path name is not set")

        # Write single atomic profile
        with self._path.open(mode="w", encoding="utf-8") as f:
            f.write(f"{profile.model_dump_json(indent=2)}\n")




