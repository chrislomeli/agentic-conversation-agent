from __future__ import annotations

from abc import ABC, abstractmethod

from conversation_engine.storage.graph import KnowledgeGraph

# from conversation_engine.models.query_node import GraphQueryPattern


# ── Graph Access Layer ─────────────────────────────────────────
class GraphAccessLayer(ABC):
    @abstractmethod
    def save_graph(self, project_name: str, graph: KnowledgeGraph) -> None:
        """ """

    @abstractmethod
    def get_graph(self, project_name: str) -> KnowledgeGraph:
        """ """

    @abstractmethod
    def delete_graph(self, project_name: str) -> bool:
        """ """

    @abstractmethod
    def list_projects(self) -> list[str]:
        """ """

    @abstractmethod
    def exists(self, project_name: str) -> bool:
        """ """
