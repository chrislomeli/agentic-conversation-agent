"""
Graph query operations for common patterns.

This module provides high-level query operations built on top of the
core graph storage. These operations implement common graph analysis patterns.
"""

from __future__ import annotations

from conversation_engine.models.base import BaseNode, EdgeType, NodeType
from conversation_engine.storage.graph import KnowledgeGraph


class GraphQueries:
    """
    High-level graph query operations.

    These queries implement common patterns for analyzing the knowledge graph:
    - Finding neighbors (nodes connected by edges)
    - Path traversal (following edge chains)
    - Orphan detection (nodes with no edges)
    - Reachability analysis (can we get from A to B?)
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    # ── Neighbor Queries ─────────────────────────────────────────────

    def get_neighbors_out(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        target_type: NodeType | None = None,
    ) -> list[BaseNode]:
        """
        Get all nodes reachable via outgoing edges.

        Args:
            node_id: Source node ID
            edge_type: Optional filter by edge type
            target_type: Optional filter by target node type

        Returns:
            List of neighbor nodes
        """
        edges = self.graph.get_outgoing_edges(node_id, edge_type)
        neighbors = []

        for edge in edges:
            target = self.graph.get_node(edge.target_id)
            if target is None:
                continue

            if target_type is not None:
                if self.graph._get_node_type(target) != target_type:
                    continue

            neighbors.append(target)

        return neighbors

    def get_neighbors_in(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        source_type: NodeType | None = None,
    ) -> list[BaseNode]:
        """
        Get all nodes that point to this node via incoming edges.

        Args:
            node_id: Target node ID
            edge_type: Optional filter by edge type
            source_type: Optional filter by source node type

        Returns:
            List of neighbor nodes
        """
        edges = self.graph.get_incoming_edges(node_id, edge_type)
        neighbors = []

        for edge in edges:
            source = self.graph.get_node(edge.source_id)
            if source is None:
                continue

            if source_type is not None:
                if self.graph._get_node_type(source) != source_type:
                    continue

            neighbors.append(source)

        return neighbors

    # ── Orphan Detection ─────────────────────────────────────────────

    def find_orphans(self, node_type: NodeType, direction: str = "out") -> list[BaseNode]:
        """
        Find nodes with no edges in the specified direction.

        Args:
            node_type: Type of nodes to check
            direction: "out" for no outgoing edges, "in" for no incoming edges

        Returns:
            List of orphan nodes
        """
        nodes = self.graph.get_nodes_by_type(node_type)
        orphans = []

        for node in nodes:
            if direction == "out":
                if self.graph.get_out_degree(node.id) == 0:
                    orphans.append(node)
            elif direction == "in":
                if self.graph.get_in_degree(node.id) == 0:
                    orphans.append(node)
            else:
                raise ValueError(f"Invalid direction: {direction}")

        return orphans

    def find_nodes_missing_edge_type(
        self, node_type: NodeType, edge_type: EdgeType, direction: str = "out"
    ) -> list[BaseNode]:
        """
        Find nodes that lack a specific edge type.

        This is useful for gap detection (e.g., goals with no SATISFIED_BY edges).

        Args:
            node_type: Type of nodes to check
            edge_type: Edge type to look for
            direction: "out" for outgoing edges, "in" for incoming edges

        Returns:
            List of nodes missing the specified edge type
        """
        nodes = self.graph.get_nodes_by_type(node_type)
        missing = []

        for node in nodes:
            if direction == "out":
                edges = self.graph.get_outgoing_edges(node.id, edge_type)
            elif direction == "in":
                edges = self.graph.get_incoming_edges(node.id, edge_type)
            else:
                raise ValueError(f"Invalid direction: {direction}")

            if len(edges) == 0:
                missing.append(node)

        return missing

    # ── Path Traversal ───────────────────────────────────────────────

    def traverse_path(
        self, start_node_id: str, edge_types: list[EdgeType], max_depth: int | None = None
    ) -> list[list[BaseNode]]:
        """
        Traverse a path following a sequence of edge types.

        This implements path pattern matching (e.g., Goal → Requirement → Step).

        Args:
            start_node_id: Starting node ID
            edge_types: Sequence of edge types to follow
            max_depth: Optional maximum depth (defaults to len(edge_types))

        Returns:
            List of paths, where each path is a list of nodes
        """
        if max_depth is None:
            max_depth = len(edge_types)

        start_node = self.graph.get_node(start_node_id)
        if start_node is None:
            return []

        paths = [[start_node]]

        for depth, edge_type in enumerate(edge_types):
            if depth >= max_depth:
                break

            new_paths = []

            for path in paths:
                current_node = path[-1]
                neighbors = self.get_neighbors_out(current_node.id, edge_type)

                for neighbor in neighbors:
                    new_path = path + [neighbor]
                    new_paths.append(new_path)

            paths = new_paths

            if not paths:
                break

        return paths

    def find_reachable_nodes(
        self, start_node_id: str, edge_types: list[EdgeType] | None = None, max_depth: int = 10
    ) -> set[str]:
        """
        Find all nodes reachable from a starting node.

        This implements breadth-first traversal.

        Args:
            start_node_id: Starting node ID
            edge_types: Optional list of edge types to follow (None = all types)
            max_depth: Maximum traversal depth

        Returns:
            Set of reachable node IDs
        """
        visited = set()
        queue = [(start_node_id, 0)]

        while queue:
            node_id, depth = queue.pop(0)

            if node_id in visited:
                continue
            if depth > max_depth:
                continue

            visited.add(node_id)

            # Get outgoing edges
            if edge_types is None:
                edges = self.graph.get_outgoing_edges(node_id)
            else:
                edges = []
                for et in edge_types:
                    edges.extend(self.graph.get_outgoing_edges(node_id, et))

            # Add neighbors to queue
            for edge in edges:
                if edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1))

        return visited

    # ── Coverage Analysis ────────────────────────────────────────────

    def get_coverage_ratio(
        self, source_type: NodeType, edge_type: EdgeType, direction: str = "out"
    ) -> float:
        """
        Calculate what fraction of nodes have at least one edge of the specified type.

        This is useful for metrics like "what % of goals have requirements?"

        Args:
            source_type: Type of nodes to check
            edge_type: Edge type to look for
            direction: "out" for outgoing edges, "in" for incoming edges

        Returns:
            Ratio between 0.0 and 1.0
        """
        nodes = self.graph.get_nodes_by_type(source_type)
        if not nodes:
            return 0.0

        with_edges = 0

        for node in nodes:
            if direction == "out":
                edges = self.graph.get_outgoing_edges(node.id, edge_type)
            else:
                edges = self.graph.get_incoming_edges(node.id, edge_type)

            if edges:
                with_edges += 1

        return with_edges / len(nodes)
