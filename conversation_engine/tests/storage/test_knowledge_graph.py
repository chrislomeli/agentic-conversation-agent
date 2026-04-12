"""
Tests for KnowledgeGraph core storage.

These tests validate proper graph semantics:
- Nodes and edges as separate entities
- Edge queryability and indexing
- Graph operations (add, remove, query)
- Transactional semantics
"""

import pytest

from conversation_engine.models import Goal, Requirement, Step
from conversation_engine.models.base import BaseEdge
from conversation_engine.storage import KnowledgeGraph


class TestNodeOperations:
    """Test node CRUD operations."""

    def test_add_and_get_node(self):
        """Test adding and retrieving a node."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Test Goal", statement="A test goal")

        graph.add_node(goal)

        retrieved = graph.get_node("goal-1")
        assert retrieved is not None
        assert retrieved.id == "goal-1"
        assert retrieved.name == "Test Goal"

    def test_add_node_replaces_existing(self):
        """Test that adding a node with same ID replaces the old one."""
        graph = KnowledgeGraph()

        goal_v1 = Goal(id="goal-1", name="Version 1", statement="V1")
        goal_v2 = Goal(id="goal-1", name="Version 2", statement="V2")

        graph.add_node(goal_v1)
        graph.add_node(goal_v2)

        retrieved = graph.get_node("goal-1")
        assert retrieved.name == "Version 2"
        assert graph.node_count() == 1

    def test_get_node_typed(self):
        """Test typed node retrieval."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        # Correct type
        retrieved_goal = graph.get_node_typed("goal-1", Goal)
        assert retrieved_goal is not None
        assert isinstance(retrieved_goal, Goal)

        # Wrong type
        retrieved_as_req = graph.get_node_typed("goal-1", Requirement)
        assert retrieved_as_req is None

    def test_remove_node_without_edges(self):
        """Test removing a node with no edges."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        graph.add_node(goal)

        result = graph.remove_node("goal-1")
        assert result is True
        assert graph.get_node("goal-1") is None
        assert graph.node_count() == 0

    def test_remove_node_with_edges_fails(self):
        """Test that removing a node with edges raises an error."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        graph.add_edge(edge)

        with pytest.raises(ValueError, match="node has edges"):
            graph.remove_node("goal-1")

    def test_remove_node_cascade(self):
        """Test removing a node and all its edges."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        graph.add_edge(edge)

        result = graph.remove_node_cascade("goal-1")
        assert result is True
        assert graph.get_node("goal-1") is None
        assert graph.edge_count() == 0

    def test_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        graph = KnowledgeGraph()

        goal1 = Goal(id="goal-1", name="Goal 1", statement="Test")
        goal2 = Goal(id="goal-2", name="Goal 2", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal1)
        graph.add_node(goal2)
        graph.add_node(req)

        goals = graph.get_nodes_by_type("goal")
        assert len(goals) == 2
        assert all(isinstance(g, Goal) for g in goals)

        reqs = graph.get_nodes_by_type("requirement")
        assert len(reqs) == 1


class TestEdgeOperations:
    """Test edge CRUD operations."""

    def test_add_and_get_edge(self):
        """Test adding and retrieving an edge."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        graph.add_edge(edge)

        retrieved = graph.get_edge("goal-1", "SATISFIED_BY", "req-1")
        assert retrieved is not None
        assert retrieved.source_id == "goal-1"
        assert retrieved.target_id == "req-1"

    def test_add_edge_requires_existing_nodes(self):
        """Test that adding an edge requires both nodes to exist."""
        graph = KnowledgeGraph()

        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(edge)

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        graph.add_node(goal)

        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge(edge)

    def test_add_edge_replaces_existing(self):
        """Test that adding an edge with same (source, type, target) replaces it."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        edge1 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        edge2 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        assert graph.edge_count() == 1

    def test_remove_edge(self):
        """Test removing an edge."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        graph.add_edge(edge)

        result = graph.remove_edge("goal-1", "SATISFIED_BY", "req-1")
        assert result is True
        assert graph.get_edge("goal-1", "SATISFIED_BY", "req-1") is None
        assert graph.edge_count() == 0

    def test_get_outgoing_edges(self):
        """Test retrieving outgoing edges."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req1 = Requirement(id="req-1", name="Requirement 1")
        req2 = Requirement(id="req-2", name="Requirement 2")

        graph.add_node(goal)
        graph.add_node(req1)
        graph.add_node(req2)

        edge1 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        edge2 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-2")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        outgoing = graph.get_outgoing_edges("goal-1")
        assert len(outgoing) == 2

        outgoing_filtered = graph.get_outgoing_edges("goal-1", "SATISFIED_BY")
        assert len(outgoing_filtered) == 2

    def test_get_incoming_edges(self):
        """Test retrieving incoming edges."""
        graph = KnowledgeGraph()

        goal1 = Goal(id="goal-1", name="Goal 1", statement="Test")
        goal2 = Goal(id="goal-2", name="Goal 2", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal1)
        graph.add_node(goal2)
        graph.add_node(req)

        edge1 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        edge2 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-2", target_id="req-1")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        incoming = graph.get_incoming_edges("req-1")
        assert len(incoming) == 2

        incoming_filtered = graph.get_incoming_edges("req-1", "SATISFIED_BY")
        assert len(incoming_filtered) == 2

    def test_get_edges_by_type(self):
        """Test retrieving all edges of a specific type."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")
        step = Step(id="step-1", name="Step")

        graph.add_node(goal)
        graph.add_node(req)
        graph.add_node(step)

        edge1 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        edge2 = BaseEdge(edge_type="REALIZED_BY", source_id="req-1", target_id="step-1")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        satisfied_by_edges = graph.get_edges_by_type("SATISFIED_BY")
        assert len(satisfied_by_edges) == 1
        assert satisfied_by_edges[0].edge_type == "SATISFIED_BY"


class TestGraphMetrics:
    """Test graph metrics and degree calculations."""

    def test_out_degree(self):
        """Test calculating out-degree of a node."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req1 = Requirement(id="req-1", name="Requirement 1")
        req2 = Requirement(id="req-2", name="Requirement 2")

        graph.add_node(goal)
        graph.add_node(req1)
        graph.add_node(req2)

        edge1 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        edge2 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-2")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        assert graph.get_out_degree("goal-1") == 2
        assert graph.get_out_degree("goal-1", "SATISFIED_BY") == 2
        assert graph.get_out_degree("req-1") == 0

    def test_in_degree(self):
        """Test calculating in-degree of a node."""
        graph = KnowledgeGraph()

        goal1 = Goal(id="goal-1", name="Goal 1", statement="Test")
        goal2 = Goal(id="goal-2", name="Goal 2", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal1)
        graph.add_node(goal2)
        graph.add_node(req)

        edge1 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        edge2 = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-2", target_id="req-1")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        assert graph.get_in_degree("req-1") == 2
        assert graph.get_in_degree("req-1", "SATISFIED_BY") == 2
        assert graph.get_in_degree("goal-1") == 0

    def test_node_and_edge_count(self):
        """Test counting nodes and edges."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        assert graph.node_count() == 0
        assert graph.edge_count() == 0

        graph.add_node(goal)
        graph.add_node(req)

        assert graph.node_count() == 2
        assert graph.edge_count() == 0

        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        graph.add_edge(edge)

        assert graph.node_count() == 2
        assert graph.edge_count() == 1


class TestGraphSemantics:
    """Test that graph maintains proper graph semantics."""

    def test_edges_independent_of_nodes(self):
        """Test that edges are separate from nodes and can be added/removed independently."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)

        # Add edge
        edge = BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1")
        graph.add_edge(edge)

        # Update node content (should not affect edge)
        updated_goal = Goal(id="goal-1", name="Updated Goal", statement="Updated")
        graph.add_node(updated_goal)

        # Edge should still exist
        retrieved_edge = graph.get_edge("goal-1", "SATISFIED_BY", "req-1")
        assert retrieved_edge is not None

        # Remove edge (should not affect nodes)
        graph.remove_edge("goal-1", "SATISFIED_BY", "req-1")

        # Nodes should still exist
        assert graph.get_node("goal-1") is not None
        assert graph.get_node("req-1") is not None

    def test_multiple_edge_types_between_same_nodes(self):
        """Test that multiple edge types can exist between the same nodes."""
        graph = KnowledgeGraph()

        node1 = Step(id="step-1", name="Step 1")
        node2 = Step(id="step-2", name="Step 2")

        graph.add_node(node1)
        graph.add_node(node2)

        edge1 = BaseEdge(edge_type="DEPENDS_ON", source_id="step-1", target_id="step-2")
        edge2 = BaseEdge(edge_type="INFORMS", source_id="step-1", target_id="step-2")

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        assert graph.edge_count() == 2
        assert graph.get_edge("step-1", "DEPENDS_ON", "step-2") is not None
        assert graph.get_edge("step-1", "INFORMS", "step-2") is not None
