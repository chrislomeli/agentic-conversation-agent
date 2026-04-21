import logging
from pathlib import Path

from journal_agent.model.session import UserProfile
from journal_agent.storage.utils import resolve_project_root

logger = logging.getLogger(__name__)


class UserProfileStore:
    """Local ProfileStore: single JSON file per user.

    Satisfies the ``ProfileStore`` protocol.  The ``user_id`` parameter
    is accepted for forward compatibility but currently ignored (single-file
    store at ``data/profile/profile.json``).
    """
    _path: Path
    def __init__(self):
        self._path = resolve_project_root() / "data" / "profile" / "profile.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_profile(self, user_id: str | None = None) -> UserProfile | None:
        try:
            with self._path.open(mode="r", encoding="utf-8") as f:
                d = f.read()
                user_profile = UserProfile.model_validate_json(d)
                user_profile.is_current = True
                user_profile.is_updated = False
                return user_profile
        except FileNotFoundError:
            return None
        except Exception:
            logger.exception("Unexpected error loading profile from %s", self._path)
            return None


    def save_profile(self, profile: UserProfile) -> None:
        # Write single atomic profile
        with self._path.open(mode="w", encoding="utf-8") as f:
            f.write(f"{profile.model_dump_json(indent=2)}\n")




