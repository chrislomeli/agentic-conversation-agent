"""
Project-level CRUD for domain configurations.

A ``ProjectStore`` persists complete ``DomainConfig`` objects keyed by
project name.  The in-memory implementation (``InMemoryProjectStore``)
is suitable for tests and local development.  Swap it for a database-
backed implementation later without changing any call site.

Usage::

    from conversation_engine.storage.project_store import InMemoryProjectStore
    from conversation_engine.models.domain_config import DomainConfig

    store = InMemoryProjectStore()
    store.save(DomainConfig(project_name="acme", project_spec=spec, rules=rules))
    cfg = store.load("acme")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from conversation_engine.models.domain_config import DomainConfig


# ── Abstract interface ───────────────────────────────────────────────


class ProjectStore(ABC):
    """
    Abstract CRUD interface for project domain configurations.

    All implementations must satisfy these semantics:
    - ``save`` is an upsert (create or overwrite).
    - ``load`` returns ``None`` when the project does not exist.
    - ``delete`` returns ``True`` when something was removed.
    - ``list_projects`` returns names in no guaranteed order.
    """

    @abstractmethod
    def save(self, config: DomainConfig) -> None:
        """
        Persist a domain configuration.

        If a project with the same ``project_name`` already exists it is
        replaced entirely.

        Args:
            config: The configuration to store.
        """
        ...

    @abstractmethod
    def load(self, project_name: str) -> Optional[DomainConfig]:
        """
        Load a domain configuration by project name.

        Args:
            project_name: The unique project identifier.

        Returns:
            The stored configuration, or ``None`` if not found.
        """
        ...

    @abstractmethod
    def delete(self, project_name: str) -> bool:
        """
        Delete a domain configuration.

        Args:
            project_name: The unique project identifier.

        Returns:
            ``True`` if the project existed and was removed,
            ``False`` if it did not exist.
        """
        ...

    @abstractmethod
    def list_projects(self) -> List[str]:
        """
        Return the names of all stored projects.

        Returns:
            A list of project names (order is not guaranteed).
        """
        ...

    @abstractmethod
    def exists(self, project_name: str) -> bool:
        """
        Check whether a project exists in the store.

        Args:
            project_name: The unique project identifier.

        Returns:
            ``True`` if the project is stored.
        """
        ...


# ── In-memory implementation ─────────────────────────────────────────


class InMemoryProjectStore(ProjectStore):
    """
    In-memory project store backed by a plain ``dict``.

    This implementation is suitable for tests and single-process use.
    It performs a shallow copy on save so that later mutations of the
    caller's objects do not silently alter stored data.
    """

    def __init__(self) -> None:
        self._store: Dict[str, DomainConfig] = {}

    def save(self, config: DomainConfig) -> None:
        if not config.project_name:
            raise ValueError("DomainConfig.project_name must not be empty")
        self._store[config.project_name] = config

    def load(self, project_name: str) -> Optional[DomainConfig]:
        return self._store.get(project_name)

    def delete(self, project_name: str) -> bool:
        if project_name in self._store:
            del self._store[project_name]
            return True
        return False

    def list_projects(self) -> List[str]:
        return list(self._store.keys())

    def exists(self, project_name: str) -> bool:
        return project_name in self._store
