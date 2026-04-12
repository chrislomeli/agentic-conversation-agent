"""
Tests for RuleEvaluator.

Validates that integrity rules are correctly evaluated against graph state.
Also validates that the evaluator accepts ProjectSpecification directly.
"""

from conversation_engine.models import Goal, Requirement, Step
from conversation_engine.models.base import BaseEdge
from conversation_engine.models.project_spec import ProjectSpecification, GoalSpec, RequirementSpec
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.storage import KnowledgeGraph
from conversation_engine.validation import RuleEvaluator
from conversation_engine.fixtures import (
    create_graph_with_gaps,
    create_graph_complete,
    create_graph_partial_coverage,
)


class TestRuleEvaluatorBasics:
    """Test basic rule evaluation."""

    def test_evaluate_rule_no_violations(self):
        """Test evaluating a rule when all nodes satisfy it."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)
        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 0

    def test_evaluate_rule_with_violations(self):
        """Test evaluating a rule when some nodes violate it."""
        graph = KnowledgeGraph()

        goal1 = Goal(id="goal-1", name="Goal 1", statement="Test")
        goal2 = Goal(id="goal-2", name="Goal 2", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal1)
        graph.add_node(goal2)
        graph.add_node(req)

        # Only goal-1 has requirement
        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 1
        assert violations[0].node_id == "goal-2"
        assert violations[0].node_name == "Goal 2"
        assert violations[0].severity == "high"
        assert "Goal 2" in violations[0].message
        assert violations[0].expected_count == 1
        assert violations[0].actual_count == 0


class TestMinimumOutgoingEdgeCount:
    """Test minimum_outgoing_edge_count rule type."""

    def test_minimum_count_satisfied(self):
        """Test when minimum count is satisfied."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req1 = Requirement(id="req-1", name="Requirement 1")
        req2 = Requirement(id="req-2", name="Requirement 2")

        graph.add_node(goal)
        graph.add_node(req1)
        graph.add_node(req2)

        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))
        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-2"))

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have at least 1 requirement",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 0

    def test_minimum_count_violated(self):
        """Test when minimum count is not satisfied."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        graph.add_node(goal)

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have at least 2 requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=2,
            severity="medium",
            failure_message_template="Goal '{subject_name}' needs more requirements.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 1
        assert violations[0].expected_count == 2
        assert violations[0].actual_count == 0


class TestExactOutgoingEdgeCount:
    """Test exact_outgoing_edge_count rule type."""

    def test_exact_count_satisfied(self):
        """Test when exact count is satisfied."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req = Requirement(id="req-1", name="Requirement")

        graph.add_node(goal)
        graph.add_node(req)
        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have exactly 1 requirement",
            applies_to_node_type="goal",
            rule_type="exact_outgoing_edge_count",
            target_node_types=["requirement"],
            exact_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' must have exactly 1 requirement.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 0

    def test_exact_count_too_many(self):
        """Test when there are too many edges."""
        graph = KnowledgeGraph()

        goal = Goal(id="goal-1", name="Goal", statement="Test")
        req1 = Requirement(id="req-1", name="Requirement 1")
        req2 = Requirement(id="req-2", name="Requirement 2")

        graph.add_node(goal)
        graph.add_node(req1)
        graph.add_node(req2)

        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-1"))
        graph.add_edge(BaseEdge(edge_type="SATISFIED_BY", source_id="goal-1", target_id="req-2"))

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have exactly 1 requirement",
            applies_to_node_type="goal",
            rule_type="exact_outgoing_edge_count",
            target_node_types=["requirement"],
            exact_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' must have exactly 1 requirement.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 1
        assert violations[0].expected_count == 1
        assert violations[0].actual_count == 2


class TestMinimumOutgoingOrFlag:
    """Test minimum_outgoing_edge_count_or_flag rule type."""

    def test_flag_allows_no_edges(self):
        """Test that explicit flag allows node to have no edges."""
        graph = KnowledgeGraph()

        step = Step(id="step-1", name="Step", has_no_dependencies=True)
        graph.add_node(step)

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Steps must declare dependencies",
            applies_to_node_type="step",
            rule_type="minimum_outgoing_edge_count_or_flag",
            target_node_types=["dependency"],
            minimum_count=1,
            allow_explicit_none_flag=True,
            severity="medium",
            failure_message_template="Step '{subject_name}' has no dependencies.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 0

    def test_no_flag_requires_edges(self):
        """Test that without flag, edges are required."""
        graph = KnowledgeGraph()

        step = Step(id="step-1", name="Step", has_no_dependencies=False)
        graph.add_node(step)

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Steps must declare dependencies",
            applies_to_node_type="step",
            rule_type="minimum_outgoing_edge_count_or_flag",
            target_node_types=["dependency"],
            minimum_count=1,
            allow_explicit_none_flag=True,
            severity="medium",
            failure_message_template="Step '{subject_name}' has no dependencies.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 1


class TestEvaluateMultipleRules:
    """Test evaluating multiple rules."""

    def test_evaluate_all_rules(self):
        """Test evaluating multiple rules at once."""
        graph = create_graph_with_gaps()

        rule1 = IntegrityRule(
            id="rule-goal",
            name="Goal Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        rule2 = IntegrityRule(
            id="rule-req",
            name="Requirement Rule",
            description="Requirements must have steps",
            applies_to_node_type="requirement",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["step"],
            minimum_count=1,
            severity="high",
            failure_message_template="Requirement '{subject_name}' has no steps.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_all_rules([rule1, rule2])

        # Should find violations for both rules
        assert len(violations) > 0

        # Check we have violations from both rules
        rule_ids = {v.rule_id for v in violations}
        assert "rule-goal" in rule_ids
        assert "rule-req" in rule_ids

    def test_get_violations_by_severity(self):
        """Test filtering violations by severity."""
        graph = create_graph_with_gaps()

        rule_high = IntegrityRule(
            id="rule-high",
            name="High Severity Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        rule_medium = IntegrityRule(
            id="rule-medium",
            name="Medium Severity Rule",
            description="Requirements must have steps",
            applies_to_node_type="requirement",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["step"],
            minimum_count=1,
            severity="medium",
            failure_message_template="Requirement '{subject_name}' has no steps.",
        )

        evaluator = RuleEvaluator(graph)

        high_violations = evaluator.get_violations_by_severity([rule_high, rule_medium], "high")
        assert all(v.severity == "high" for v in high_violations)

        medium_violations = evaluator.get_violations_by_severity([rule_high, rule_medium], "medium")
        assert all(v.severity == "medium" for v in medium_violations)

    def test_get_violations_for_node(self):
        """Test getting violations for a specific node."""
        graph = create_graph_with_gaps()

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        evaluator = RuleEvaluator(graph)

        # goal-2 should have violations
        violations = evaluator.get_violations_for_node("goal-2", [rule])
        assert len(violations) > 0
        assert all(v.node_id == "goal-2" for v in violations)

        # goal-1 should have no violations
        violations = evaluator.get_violations_for_node("goal-1", [rule])
        assert len(violations) == 0


class TestWithFixtures:
    """Test evaluator with fixture graphs."""

    def test_complete_graph_no_violations(self):
        """Test that complete graph has no violations."""
        graph = create_graph_complete()

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        assert len(violations) == 0

    def test_partial_coverage_graph(self):
        """Test graph with partial coverage."""
        graph = create_graph_partial_coverage()

        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="Goals must have requirements",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )

        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(rule)

        # Should have 2 violations (goal-2 and goal-3)
        assert len(violations) == 2


class TestEvaluatorWithProjectSpecification:
    """Test that RuleEvaluator accepts ProjectSpecification directly."""

    _goal_req_rule = IntegrityRule(
        id="rule-goal-req",
        name="Goal → Requirement",
        description="Every goal must have at least one requirement",
        applies_to_node_type="goal",
        rule_type="minimum_outgoing_edge_count",
        target_node_types=["requirement"],
        minimum_count=1,
        severity="high",
        failure_message_template="Goal '{subject_name}' has no requirements.",
    )

    def test_spec_no_violations(self):
        """Complete spec produces no violations."""
        spec = ProjectSpecification(
            project_name="ok",
            goals=[GoalSpec(name="G1", statement="A goal")],
            requirements=[RequirementSpec(name="R1", goal_ref="G1")],
        )
        evaluator = RuleEvaluator(spec)
        violations = evaluator.evaluate_rule(self._goal_req_rule)
        assert len(violations) == 0

    def test_spec_with_violations(self):
        """Spec with orphan goal produces a violation."""
        spec = ProjectSpecification(
            project_name="gaps",
            goals=[
                GoalSpec(name="Connected", statement="Has a req"),
                GoalSpec(name="Orphan", statement="No req"),
            ],
            requirements=[RequirementSpec(name="R1", goal_ref="Connected")],
        )
        evaluator = RuleEvaluator(spec)
        violations = evaluator.evaluate_rule(self._goal_req_rule)
        assert len(violations) == 1
        assert "Orphan" in violations[0].message

    def test_spec_multiple_rules(self):
        """Multiple rules evaluated against a spec."""
        req_step_rule = IntegrityRule(
            id="rule-req-step",
            name="Requirement → Step",
            description="Every requirement must have at least one step",
            applies_to_node_type="requirement",
            rule_type="minimum_outgoing_edge_count",
            target_node_types=["step"],
            minimum_count=1,
            severity="medium",
            failure_message_template="Requirement '{subject_name}' has no steps.",
        )
        spec = ProjectSpecification(
            project_name="multi",
            goals=[GoalSpec(name="G1", statement="A goal")],
            requirements=[RequirementSpec(name="R1", goal_ref="G1")],
        )
        evaluator = RuleEvaluator(spec)
        violations = evaluator.evaluate_all_rules([self._goal_req_rule, req_step_rule])
        # G1→R1 satisfied, but R1 has no step
        assert len(violations) == 1
        assert violations[0].rule_id == "rule-req-step"

    def test_empty_spec_no_violations(self):
        """Empty spec has no nodes, so no violations."""
        spec = ProjectSpecification(project_name="empty")
        evaluator = RuleEvaluator(spec)
        violations = evaluator.evaluate_rule(self._goal_req_rule)
        assert len(violations) == 0

    def test_graph_still_works(self):
        """Backward compat: KnowledgeGraph input still works."""
        graph = KnowledgeGraph()
        goal = Goal(id="g1", name="G1", statement="Test")
        graph.add_node(goal)
        evaluator = RuleEvaluator(graph)
        violations = evaluator.evaluate_rule(self._goal_req_rule)
        assert len(violations) == 1
