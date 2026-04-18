from pathlib import Path

from journal_agent.model.session import UserProfile
from journal_agent.storage.utils import resolve_project_root


class ProfileStore:
    path: Path
    def __init__(self):
        self._path = resolve_project_root() / "data" / "profile" / "profile.json"
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

    def load_profile(self, profile: UserProfile | None = None) -> UserProfile | None:
        if self._path is None:
            raise ValueError("Path name is not set")

        # Write single atomic profile
        with self.path.open(mode="r", encoding="utf-8") as f:
            f.read()
            return UserProfile.model_validate_json(f.read())

    def save_profile(self,  profile: UserProfile | None = None):
        if self._path is None:
            raise ValueError("Path name is not set")

        # Write single atomic profile
        with self.path.open(mode="w", encoding="utf-8") as f:
            f.write(f"{profile.model_dump_json(indent=2)}\n")




