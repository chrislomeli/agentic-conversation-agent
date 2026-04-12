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

from typing import TYPE_CHECKING

from conversation_engine.models.graph_db_access import GraphAccessLayer
from conversation_engine.storage import ProjectStore
from conversation_engine.storage.project_facade import graph_to_domain_config, project_to_graph

if TYPE_CHECKING:
    from conversation_engine.models.domain_config import DomainConfig


# ── Graph implementation ─────────────────────────────────────────


class GraphProjectStore(ProjectStore):
    """
    In-memory project store backed by a plain ``dict``.

    This implementation is suitable for tests and single-process use.
    It performs a shallow copy on save so that later mutations of the
    caller's objects do not silently alter stored data.
    """

    def __init__(self, graph: GraphAccessLayer) -> None:
        self._graph_access_layer = graph

    def save(self, config: DomainConfig) -> None:
        if not config.project_name:
            raise ValueError("DomainConfig.project_name must not be empty")

        # project_name
        project_graph = project_to_graph(config)

        # save the graph
        self._graph_access_layer.save_graph(config.project_name, project_graph)

    def load(self, project_name: str) -> DomainConfig | None:
        graph = self._graph_access_layer.get_graph(project_name)
        if graph is None:
            return None
        return graph_to_domain_config(graph)

    def delete(self, project_name: str) -> bool:
        return self._graph_access_layer.delete_graph(project_name)

    def list_projects(self) -> list[str]:
        return self._graph_access_layer.list_projects()

    def exists(self, project_name: str) -> bool:
        return self._graph_access_layer.exists(project_name)
