"""
File-backed project store — one JSON file per project.

This is an interim persistence layer that mirrors the interface a
Memgraph-backed store would expose.  ``save()`` writes the entire
``DomainConfig`` as a single JSON document; ``load()`` reads it back.

Usage::

    from conversation_engine.storage.file_project_store import FileProjectStore
    from conversation_engine.models.domain_config import DomainConfig

    store = FileProjectStore("/path/to/projects")
    store.save(config)
    loaded = store.load("my-project")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from conversation_engine.storage.project_store import ProjectStore

if TYPE_CHECKING:
    from conversation_engine.models.domain_config import DomainConfig


class FileProjectStore(ProjectStore):
    """
    Persistent project store backed by JSON files on disk.

    Each project is stored as ``<base_dir>/<project_name>.json``.
    The directory is created automatically if it does not exist.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, project_name: str) -> Path:
        return self._base_dir / f"{project_name}.json"

    def save(self, config: DomainConfig) -> None:
        if not config.project_name:
            raise ValueError("DomainConfig.project_name must not be empty")
        path = self._path_for(config.project_name)
        path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    def load(self, project_name: str) -> Optional[DomainConfig]:
        from conversation_engine.models.domain_config import DomainConfig as _DC

        path = self._path_for(project_name)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return _DC.from_dict(data)

    def delete(self, project_name: str) -> bool:
        path = self._path_for(project_name)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_projects(self) -> List[str]:
        return sorted(p.stem for p in self._base_dir.glob("*.json"))

    def exists(self, project_name: str) -> bool:
        return self._path_for(project_name).exists()
