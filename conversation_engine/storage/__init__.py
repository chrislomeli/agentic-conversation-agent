"""
Graph storage layer for the Conversation Engine.

Provides in-memory graph storage with proper graph semantics:
- Nodes and edges as first-class entities
- Queryable edge relationships
- Graph traversal operations
- Index-backed lookups
- Project-level CRUD for domain configurations
"""

from conversation_engine.models.project_spec import (
    ConstraintSpec,
    DependencySpec,
    GoalSpec,
    ProjectSnapshot,  # backwards-compatible alias
    ProjectSpecification,
    RequirementSpec,
    StepSpec,
)
from conversation_engine.storage.file_project_store import FileProjectStore
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.project_store import (
    InMemoryProjectStore,
    ProjectStore,
)
from conversation_engine.storage.queries import GraphQueries
from conversation_engine.storage.snapshot_facade import (
    SnapshotConversionError,
    graph_to_snapshot,
    snapshot_to_graph,
)

__all__ = [
    "KnowledgeGraph",
    "GraphQueries",
    "ProjectStore",
    "InMemoryProjectStore",
    "FileProjectStore",
    # Project specification
    "ProjectSpecification",
    "ProjectSnapshot",
    "GoalSpec",
    "RequirementSpec",
    "StepSpec",
    "ConstraintSpec",
    "DependencySpec",
    "snapshot_to_graph",
    "graph_to_snapshot",
    "SnapshotConversionError",
]
