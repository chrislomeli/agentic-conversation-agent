"""
Tests for graph fixtures.

Validates that fixtures create graphs with expected characteristics.
"""

from conversation_engine.fixtures import (
    create_graph_complete,
    create_graph_partial_coverage,
    create_graph_with_gaps,
    create_graph_with_orphans,
    create_minimal_graph,
)
from conversation_engine.storage import GraphQueries


class TestMinimalGraph:
    """Test minimal graph fixture."""

    def test_creates_minimal_graph(self):
        """Test that minimal graph is created correctly."""
        graph = create_minimal_graph()

        assert graph.node_count() == 2
        assert graph.edge_count() == 1

    def test_has_simple_chain(self):
        """Test that minimal graph has a simple chain."""
        graph = create_minimal_graph()
        queries = GraphQueries(graph)

        paths = queries.traverse_path("goal-1", ["SATISFIED_BY"])

        assert len(paths) == 1
        assert len(paths[0]) == 2


class TestGraphWithGaps:
    """Test graph with gaps fixture."""

    def test_creates_graph_with_gaps(self):
        """Test that graph with gaps is created."""
        graph = create_graph_with_gaps()

        # Should have multiple goals
        goals = graph.get_nodes_by_type("goal")
        assert len(goals) == 3

    def test_has_orphan_goals(self):
        """Test that some goals have no requirements."""
        graph = create_graph_with_gaps()
        queries = GraphQueries(graph)

        orphans = queries.find_nodes_missing_edge_type("goal", "SATISFIED_BY", direction="out")

        # Should have 2 orphan goals (goal-2 and goal-3)
        assert len(orphans) == 2

    def test_has_orphan_requirements(self):
        """Test that some requirements have no outgoing REALIZED_BY edges."""
        graph = create_graph_with_gaps()
        queries = GraphQueries(graph)

        orphans = queries.find_nodes_missing_edge_type(
            "requirement", "REALIZED_BY", direction="out"
        )

        # Both requirements have no REALIZED_BY edges (no components linked)
        assert len(orphans) == 2

    def test_coverage_is_partial(self):
        """Test that coverage is not 100%."""
        graph = create_graph_with_gaps()
        queries = GraphQueries(graph)

        coverage = queries.get_coverage_ratio("goal", "SATISFIED_BY", direction="out")

        # Only 1 out of 3 goals has requirements
        assert coverage < 1.0
        assert coverage > 0.0


class TestGraphWithOrphans:
    """Test graph with orphans fixture."""

    def test_creates_graph_with_orphans(self):
        """Test that graph with orphans is created."""
        graph = create_graph_with_orphans()

        assert graph.node_count() > 0

    def test_has_completely_isolated_nodes(self):
        """Test that some nodes have no edges at all."""
        graph = create_graph_with_orphans()
        GraphQueries(graph)

        # goal-2 should be completely isolated
        out_degree = graph.get_out_degree("goal-2")
        in_degree = graph.get_in_degree("goal-2")

        assert out_degree == 0
        assert in_degree == 0

    def test_has_connected_nodes(self):
        """Test that some nodes are connected."""
        graph = create_graph_with_orphans()

        # goal-1 should have outgoing edge
        out_degree = graph.get_out_degree("goal-1")
        assert out_degree > 0


class TestCompleteGraph:
    """Test complete graph fixture."""

    def test_creates_complete_graph(self):
        """Test that complete graph is created."""
        graph = create_graph_complete()

        assert graph.node_count() > 0
        assert graph.edge_count() > 0

    def test_all_goals_have_requirements(self):
        """Test that all goals have requirements."""
        graph = create_graph_complete()
        queries = GraphQueries(graph)

        coverage = queries.get_coverage_ratio("goal", "SATISFIED_BY", direction="out")

        assert coverage == 1.0

    def test_all_requirements_have_components(self):
        """Test that all requirements have components."""
        graph = create_graph_complete()
        queries = GraphQueries(graph)

        coverage = queries.get_coverage_ratio("requirement", "REALIZED_BY", direction="out")

        assert coverage == 1.0

    def test_has_complete_chains(self):
        """Test that complete traceability chains exist."""
        graph = create_graph_complete()
        queries = GraphQueries(graph)

        # Should be able to traverse from goal to component
        paths = queries.traverse_path("goal-1", ["SATISFIED_BY", "REALIZED_BY"])

        assert len(paths) > 0
        assert all(len(path) == 3 for path in paths)


class TestPartialCoverageGraph:
    """Test partial coverage graph fixture."""

    def test_creates_partial_coverage_graph(self):
        """Test that partial coverage graph is created."""
        graph = create_graph_partial_coverage()

        goals = graph.get_nodes_by_type("goal")
        assert len(goals) == 3

    def test_has_33_percent_coverage(self):
        """Test that coverage is approximately 33%."""
        graph = create_graph_partial_coverage()
        queries = GraphQueries(graph)

        coverage = queries.get_coverage_ratio("goal", "SATISFIED_BY", direction="out")

        # 1 out of 3 goals has requirements
        assert abs(coverage - 0.333) < 0.01

    def test_one_goal_has_multiple_requirements(self):
        """Test that connected goal has multiple requirements."""
        graph = create_graph_partial_coverage()

        out_degree = graph.get_out_degree("goal-1", "SATISFIED_BY")

        assert out_degree == 2

    def test_two_goals_have_no_requirements(self):
        """Test that two goals have no requirements."""
        graph = create_graph_partial_coverage()
        queries = GraphQueries(graph)

        orphans = queries.find_nodes_missing_edge_type("goal", "SATISFIED_BY", direction="out")

        assert len(orphans) == 2
