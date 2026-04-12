"""
Tests for conversation_engine Pydantic models.

Validates that models can be instantiated and serialized correctly.
"""

import pytest
from pydantic import ValidationError

from conversation_engine.models import (
    # GraphQueryPattern,
    Assessment,
    Goal,
    GoalRequirementTrace,
    IntegrityRule,
    Requirement,
    Step,
)


class TestNodeModels:
    """Test node type models."""

    def test_goal_creation(self):
        """Test Goal model instantiation."""
        goal = Goal(id="goal-test", name="Test Goal", statement="This is a test goal statement.")
        assert goal.id == "goal-test"
        assert goal.name == "Test Goal"
        assert goal.statement == "This is a test goal statement."

    def test_goal_serialization(self):
        """Test Goal model serialization."""
        goal = Goal(id="goal-test", name="Test Goal", statement="This is a test goal statement.")
        data = goal.model_dump()
        assert data["id"] == "goal-test"
        assert data["name"] == "Test Goal"

        goal_from_dict = Goal(**data)
        assert goal_from_dict.id == goal.id

    def test_requirement_creation(self):
        """Test Requirement model instantiation."""
        req = Requirement(
            id="REQ-001",
            name="Test Requirement",
            requirement_type="functional",
            description="A test requirement",
        )
        assert req.id == "REQ-001"
        assert req.requirement_type == "functional"

    def test_step_with_no_dependencies(self):
        """Test Step with explicit no dependencies flag."""
        step = Step(id="step-test", name="Test Step", has_no_dependencies=True)
        assert step.has_no_dependencies is True


class TestTraceabilityModels:
    """Test traceability models."""

    def test_goal_requirement_trace(self):
        """Test GoalRequirementTrace model."""
        trace = GoalRequirementTrace(goal_id="goal-1", requirement_ids=["REQ-001", "REQ-002"])
        assert trace.goal_id == "goal-1"
        assert len(trace.requirement_ids) == 2
        assert "REQ-001" in trace.requirement_ids

    def test_empty_trace(self):
        """Test trace with no linked items."""
        trace = GoalRequirementTrace(goal_id="goal-orphan", requirement_ids=[])
        assert len(trace.requirement_ids) == 0


class TestIntegrityRules:
    """Test integrity rule models."""

    def test_integrity_rule_creation(self):
        """Test IntegrityRule model instantiation."""
        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="A test rule",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            edge_type="SATISFIED_BY",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Goal '{subject_name}' has no requirements.",
        )
        assert rule.id == "rule-test"
        assert rule.severity == "high"
        assert rule.minimum_count == 1

    def test_integrity_rule_immutable(self):
        """Test that IntegrityRule is frozen."""
        rule = IntegrityRule(
            id="rule-test",
            name="Test Rule",
            description="A test rule",
            applies_to_node_type="goal",
            rule_type="minimum_outgoing_edge_count",
            edge_type="SATISFIED_BY",
            target_node_types=["requirement"],
            minimum_count=1,
            severity="high",
            failure_message_template="Test",
        )
        with pytest.raises(ValidationError):
            rule.severity = "low"


class TestAssessmentModels:
    """Test assessment output models."""

    def test_assessment_creation(self):
        """Test Assessment model instantiation."""
        assessment = Assessment(
            id="assessment-001",
            assessment_type="missing_goal_coverage",
            severity="high",
            subject_ids=["goal-1"],
            finding="Goal has no linked requirements.",
            evidence=["No SATISFIED_BY edges found"],
            related_rule_ids=["rule-goal-must-have-requirement"],
            suggested_actions=["Add requirements to satisfy this goal"],
            confidence="high",
        )
        assert assessment.id == "assessment-001"
        assert assessment.severity == "high"
        assert len(assessment.evidence) == 1
        assert len(assessment.suggested_actions) == 1
