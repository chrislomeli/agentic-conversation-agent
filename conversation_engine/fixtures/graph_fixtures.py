"""
Graph fixtures for testing.

These fixtures create graphs with specific characteristics for testing
AI reasoning components.
"""

from conversation_engine.models import Goal, Requirement, Step
from conversation_engine.models.base import BaseEdge
from conversation_engine.storage import KnowledgeGraph


def create_minimal_graph() -> KnowledgeGraph:
    """
    Create a minimal graph with just a few nodes and edges.

    Structure:
    - 1 Goal → 1 Requirement

    Use case: Basic smoke tests
    """
    graph = KnowledgeGraph()

    goal = Goal(id="goal-1", name="Test Goal", statement="A test goal")
    req = Requirement(id="req-1", name="Test Requirement")

    graph.add_node(goal)
    graph.add_node(req)

    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))

    return graph


def create_graph_with_gaps() -> KnowledgeGraph:
    """
    Create a graph with intentional gaps (missing edges).

    Structure:
    - Goal 1 → Requirement 1 (connected)
    - Goal 2 (no requirements - GAP)
    - Goal 3 (no requirements - GAP)
    - Requirement 2 (no components - GAP)

    Use case: Testing gap detection and AI explanation
    """
    graph = KnowledgeGraph()

    # Goals
    goal1 = Goal(id="goal-1", name="Connected Goal", statement="Has requirements")
    goal2 = Goal(id="goal-2", name="Orphan Goal", statement="Missing requirements")
    goal3 = Goal(id="goal-3", name="Another Orphan", statement="Also missing requirements")

    graph.add_node(goal1)
    graph.add_node(goal2)
    graph.add_node(goal3)

    # Requirements
    req1 = Requirement(id="req-1", name="Connected Requirement")
    req2 = Requirement(id="req-2", name="Orphan Requirement")

    graph.add_node(req1)
    graph.add_node(req2)

    # Edges (intentionally incomplete)
    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))
    # goal-2 and goal-3 have no edges (gaps)
    # req-2 has no outgoing edges (gap)

    return graph


def create_graph_with_orphans() -> KnowledgeGraph:
    """
    Create a graph with orphan nodes (no incoming or outgoing edges).

    Structure:
    - Goal 1 → Requirement 1 (connected)
    - Goal 2 (completely isolated)
    - Requirement 2 (completely isolated)

    Use case: Testing orphan detection
    """
    graph = KnowledgeGraph()

    # Connected nodes
    goal1 = Goal(id="goal-1", name="Connected Goal", statement="Has edges")
    req1 = Requirement(id="req-1", name="Connected Requirement")

    graph.add_node(goal1)
    graph.add_node(req1)

    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))

    # Orphan nodes (no edges at all)
    goal2 = Goal(id="goal-2", name="Orphan Goal", statement="No edges")
    req2 = Requirement(id="req-2", name="Orphan Requirement")

    graph.add_node(goal2)
    graph.add_node(req2)

    return graph


def create_graph_complete() -> KnowledgeGraph:
    """
    Create a complete graph with no gaps.

    Structure:
    - Goal 1 → Requirement 1 → Step 1
    - Goal 2 → Requirement 2 → Step 2
    - All nodes properly connected

    Use case: Testing that validation passes when graph is complete
    """
    graph = KnowledgeGraph()

    # Chain 1
    goal1 = Goal(id="goal-1", name="Goal 1", statement="First goal")
    req1 = Requirement(id="req-1", name="Requirement 1")
    step1 = Step(id="step-1", name="Step 1")

    graph.add_node(goal1)
    graph.add_node(req1)
    graph.add_node(step1)

    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))
    graph.add_edge(BaseEdge(edge_type="REALIZED_BY", source_id="req-1", target_id="step-1"))

    # Chain 2
    goal2 = Goal(id="goal-2", name="Goal 2", statement="Second goal")
    req2 = Requirement(id="req-2", name="Requirement 2")
    step2 = Step(id="step-2", name="Step 2")

    graph.add_node(goal2)
    graph.add_node(req2)
    graph.add_node(step2)

    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-2", target_id="req-2"))
    graph.add_edge(BaseEdge(edge_type="REALIZED_BY", source_id="req-2", target_id="step-2"))

    return graph


def create_graph_partial_coverage() -> KnowledgeGraph:
    """
    Create a graph with partial coverage (some nodes connected, some not).

    Structure:
    - 3 Goals total
    - 1 Goal has requirements (33% coverage)
    - 2 Goals missing requirements (67% gaps)

    Use case: Testing coverage metrics and AI prioritization
    """
    graph = KnowledgeGraph()

    # Goals
    goal1 = Goal(id="goal-1", name="Goal with Requirements", statement="Connected")
    goal2 = Goal(id="goal-2", name="Goal without Requirements", statement="Gap")
    goal3 = Goal(id="goal-3", name="Another Goal without Requirements", statement="Gap")

    graph.add_node(goal1)
    graph.add_node(goal2)
    graph.add_node(goal3)

    # Requirements
    req1 = Requirement(id="req-1", name="Requirement 1")
    req2 = Requirement(id="req-2", name="Requirement 2")

    graph.add_node(req1)
    graph.add_node(req2)

    # Only goal-1 has requirements
    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))
    graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-2"))

    # goal-2 and goal-3 have no requirements (partial coverage)

    return graph
