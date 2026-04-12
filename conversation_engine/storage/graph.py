"""
Core graph storage implementation.

This is a proper graph data structure with nodes and edges as separate entities.
Edges are queryable and indexable, enabling graph traversal and analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Type, TypeVar
from collections import defaultdict

from conversation_engine.models.base import BaseNode, BaseEdge, NodeType, EdgeType
from conversation_engine.models.nodes import (
    Feature,
    Goal,
    GuidingPrinciple,
    Requirement,
    Capability,
    UseCase,
    Scenario,
    DesignArtifact,
    Decision,
    Constraint,
    Component,
    Step,
    Dependency,
    DocumentationArtifact,
    Project,
)
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.models.validation_quiz import ValidationQuiz
# from conversation_engine.models.query_node import GraphQueryPattern


T = TypeVar("T", bound=BaseNode)

# Registry maps NodeType enum values → concrete Pydantic classes for deserialisation.
_NODE_TYPE_REGISTRY: Dict[NodeType, type[BaseNode]] = {
    NodeType.FEATURE: Feature,
    NodeType.GOAL: Goal,
    NodeType.GUIDING_PRINCIPLE: GuidingPrinciple,
    NodeType.REQUIREMENT: Requirement,
    NodeType.CAPABILITY: Capability,
    NodeType.USE_CASE: UseCase,
    NodeType.SCENARIO: Scenario,
    NodeType.DESIGN_ARTIFACT: DesignArtifact,
    NodeType.DECISION: Decision,
    NodeType.CONSTRAINT: Constraint,
    NodeType.COMPONENT: Component,
    NodeType.STEP: Step,
    NodeType.DEPENDENCY: Dependency,
    NodeType.DOCUMENTATION_ARTIFACT: DocumentationArtifact,
    NodeType.PROJECT: Project,
    NodeType.RULE: IntegrityRule,
    NodeType.QUIZ: ValidationQuiz,
    # NodeType.QUERY_PATTERN: GraphQueryPattern,
}


class KnowledgeGraph:
    """
    In-memory knowledge graph with proper graph semantics.

    Design principles:
    - Nodes and edges are separate entities (not embedded references)
    - Edges are first-class citizens with their own identity
    - All operations are indexed for O(1) or O(k) lookup where k = result size
    - Type-safe operations using Pydantic models
    - Transactional semantics (operations succeed or fail atomically)

    This implementation uses the same conceptual model as Neo4j or MemGraph,
    making it easy to swap for a real graph database later.
    """

    def __init__(self):
        # Node storage: id -> node
        self._nodes: Dict[str, BaseNode] = {}

        # Edge storage: (source_id, edge_type, target_id) -> edge
        # Using tuple key ensures edge uniqueness
        self._edges: Dict[tuple[str, EdgeType, str], BaseEdge] = {}

        # Indexes for efficient queries
        # Outgoing edges: source_id -> list of edges
        self._outgoing_index: Dict[str, List[BaseEdge]] = defaultdict(list)

        # Incoming edges: target_id -> list of edges
        self._incoming_index: Dict[str, List[BaseEdge]] = defaultdict(list)

        # Edge type index: edge_type -> list of edges
        self._edge_type_index: Dict[EdgeType, List[BaseEdge]] = defaultdict(list)

    # ── Node Operations ──────────────────────────────────────────────

    def add_node(self, node: BaseNode) -> None:
        """
        Add a node to the graph.

        If a node with the same ID already exists, it will be replaced.
        This allows updating node content without breaking edge references.

        Args:
            node: The node to add
        """
        # Simply add/replace the node - no index maintenance needed
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """
        Retrieve a node by ID.

        Args:
            node_id: The node identifier

        Returns:
            The node if found, None otherwise
        """
        return self._nodes.get(node_id)

    def get_node_typed(self, node_id: str, node_class: Type[T]) -> Optional[T]:
        """
        Retrieve a node by ID with type checking.

        Args:
            node_id: The node identifier
            node_class: Expected node class (e.g., Goal, Requirement)

        Returns:
            The node if found and of correct type, None otherwise
        """
        node = self._nodes.get(node_id)
        if node is None:
            return None
        if not isinstance(node, node_class):
            return None
        return node

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the graph.

        This will fail if the node has any edges (incoming or outgoing).
        Use remove_node_cascade() to remove edges automatically.

        Args:
            node_id: The node identifier

        Returns:
            True if removed, False if node doesn't exist

        Raises:
            ValueError: If node has edges
        """
        if node_id not in self._nodes:
            return False

        # Check for edges
        if self._outgoing_index[node_id] or self._incoming_index[node_id]:
            raise ValueError(
                f"Cannot remove node {node_id}: node has edges. "
                "Use remove_node_cascade() to remove edges automatically."
            )

        # Remove from storage - no index maintenance needed
        del self._nodes[node_id]

        return True

    def remove_node_cascade(self, node_id: str) -> bool:
        """
        Remove a node and all its edges from the graph.

        Args:
            node_id: The node identifier

        Returns:
            True if removed, False if node doesn't exist
        """
        if node_id not in self._nodes:
            return False

        # Remove all edges
        outgoing = list(self._outgoing_index[node_id])
        incoming = list(self._incoming_index[node_id])

        for edge in outgoing + incoming:
            self.remove_edge(edge.source_id, edge.edge_type, edge.target_id)

        # Now remove the node
        return self.remove_node(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNode]:
        """
        Get all nodes of a specific type using enum traversal.

        Args:
            node_type: The node type to filter by

        Returns:
            List of nodes of the specified type
        """
        return [node for node in self._nodes.values() if node.node_type == node_type]

    def get_all_nodes(self) -> List[BaseNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())

    # ── Edge Operations ──────────────────────────────────────────────

    def add_edge(self, edge: BaseEdge) -> None:
        """
        Add an edge to the graph.

        Both source and target nodes must exist in the graph.
        If an edge with the same (source, type, target) exists, it will be replaced.

        Args:
            edge: The edge to add

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} does not exist")

        edge_key = (edge.source_id, edge.edge_type, edge.target_id)

        # Remove old edge from indexes if it exists
        if edge_key in self._edges:
            self._remove_edge_from_indexes(self._edges[edge_key])

        # Add new edge
        self._edges[edge_key] = edge

        # Update indexes
        self._outgoing_index[edge.source_id].append(edge)
        self._incoming_index[edge.target_id].append(edge)
        self._edge_type_index[edge.edge_type].append(edge)

    def get_edge(self, source_id: str, edge_type: EdgeType, target_id: str) -> Optional[BaseEdge]:
        """
        Retrieve a specific edge.

        Args:
            source_id: Source node ID
            edge_type: Edge type
            target_id: Target node ID

        Returns:
            The edge if found, None otherwise
        """
        edge_key = (source_id, edge_type, target_id)
        return self._edges.get(edge_key)

    def remove_edge(self, source_id: str, edge_type: EdgeType, target_id: str) -> bool:
        """
        Remove an edge from the graph.

        Args:
            source_id: Source node ID
            edge_type: Edge type
            target_id: Target node ID

        Returns:
            True if removed, False if edge doesn't exist
        """
        edge_key = (source_id, edge_type, target_id)
        edge = self._edges.get(edge_key)

        if edge is None:
            return False

        # Remove from storage and indexes
        del self._edges[edge_key]
        self._remove_edge_from_indexes(edge)

        return True

    def get_outgoing_edges(
        self, source_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[BaseEdge]:
        """
        Get all outgoing edges from a node.

        Args:
            source_id: Source node ID
            edge_type: Optional filter by edge type

        Returns:
            List of outgoing edges
        """
        edges = self._outgoing_index.get(source_id, [])

        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]

        return edges

    def get_incoming_edges(
        self, target_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[BaseEdge]:
        """
        Get all incoming edges to a node.

        Args:
            target_id: Target node ID
            edge_type: Optional filter by edge type

        Returns:
            List of incoming edges
        """
        edges = self._incoming_index.get(target_id, [])

        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]

        return edges

    def get_edges_by_type(self, edge_type: EdgeType) -> List[BaseEdge]:
        """
        Get all edges of a specific type.

        Args:
            edge_type: The edge type to filter by

        Returns:
            List of edges of the specified type
        """
        return list(self._edge_type_index.get(edge_type, []))

    def get_all_edges(self) -> List[BaseEdge]:
        """Get all edges in the graph."""
        return list(self._edges.values())

    # ── Graph Metrics ────────────────────────────────────────────────

    def get_out_degree(self, node_id: str, edge_type: Optional[EdgeType] = None) -> int:
        """
        Get the out-degree of a node (number of outgoing edges).

        Args:
            node_id: Node identifier
            edge_type: Optional filter by edge type

        Returns:
            Number of outgoing edges
        """
        return len(self.get_outgoing_edges(node_id, edge_type))

    def get_in_degree(self, node_id: str, edge_type: Optional[EdgeType] = None) -> int:
        """
        Get the in-degree of a node (number of incoming edges).

        Args:
            node_id: Node identifier
            edge_type: Optional filter by edge type

        Returns:
            Number of incoming edges
        """
        return len(self.get_incoming_edges(node_id, edge_type))

    def node_count(self) -> int:
        """Get total number of nodes in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get total number of edges in the graph."""
        return len(self._edges)

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise the full graph to a plain dict.

        The output is JSON-safe (no custom objects) and suitable for
        persistence.  Mirrors what a ``MATCH (n)-[r]->(m)`` export
        from Memgraph would produce.

        Returns:
            A dict with ``"nodes"`` and ``"edges"`` lists.
        """
        nodes = []
        for node in self._nodes.values():
            data = node.model_dump()
            data["_type"] = self._get_node_type(node)
            nodes.append(data)

        edges = [e.model_dump() for e in self._edges.values()]

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """
        Reconstruct a KnowledgeGraph from the dict produced by ``to_dict()``.

        Raises:
            ValueError: If a node ``_type`` is not in the registry.
        """
        graph = cls()

        for node_data in data.get("nodes", []):
            node_type_str = node_data.pop("_type", None)
            if node_type_str is None:
                raise ValueError(f"Node data missing '_type': {node_data}")

            # Convert string back to NodeType enum
            node_type = NodeType(node_type_str)
            node_cls = _NODE_TYPE_REGISTRY.get(node_type)
            if node_cls is None:
                raise ValueError(
                    f"Unknown node type '{node_type}'. Known types: {sorted(_NODE_TYPE_REGISTRY)}"
                )
            graph.add_node(node_cls.model_validate(node_data))

        for edge_data in data.get("edges", []):
            graph.add_edge(BaseEdge.model_validate(edge_data))

        return graph

    # ── Internal Helpers ─────────────────────────────────────────────

    def _get_node_type(self, node: BaseNode) -> NodeType:
        """
        Get node type from the node's node_type field.

        This is much simpler than inferring from class names.
        """
        return node.node_type

    def _remove_edge_from_indexes(self, edge: BaseEdge) -> None:
        """Remove an edge from all indexes."""
        self._outgoing_index[edge.source_id].remove(edge)
        self._incoming_index[edge.target_id].remove(edge)
        self._edge_type_index[edge.edge_type].remove(edge)
