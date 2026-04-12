"""
Integrity rule evaluator.

This module evaluates integrity rules against graph state and returns violations.
This is deterministic validation - no AI involved.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from conversation_engine.models.base import BaseNode
from conversation_engine.models.project_spec import ProjectSpecification
from conversation_engine.models.rule_node import IntegrityRule, Severity
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.snapshot_facade import snapshot_to_graph


class RuleViolation(BaseModel):
    """
    A violation of an integrity rule.

    This represents a specific instance where a node fails to satisfy a rule.
    """

    rule_id: str = Field(..., description="ID of the violated rule")
    node_id: str = Field(..., description="ID of the node that violates the rule")
    node_name: str = Field(..., description="Name of the node for human readability")
    severity: Severity = Field(..., description="Severity level of the violation")
    message: str = Field(..., description="Human-readable violation message")

    expected_count: int | None = Field(None, description="Expected edge count (for context)")
    actual_count: int = Field(..., description="Actual edge count found")


class RuleEvaluator:
    """
    Evaluates integrity rules against graph state.

    This is deterministic validation that checks graph structure against rules
    and returns violations. The AI will use these violations as input for reasoning.

    Design principles:
    - Deterministic: same graph + same rules = same violations
    - No AI: pure logic, no LLM calls
    - Fast: O(n) where n = nodes of target type
    - Clear output: violations are actionable
    """

    def __init__(self, graph_or_spec: KnowledgeGraph | ProjectSpecification):
        if isinstance(graph_or_spec, ProjectSpecification):
            self.graph = snapshot_to_graph(graph_or_spec)
        else:
            self.graph = graph_or_spec

    def evaluate_rule(self, rule: IntegrityRule) -> list[RuleViolation]:
        """
        Evaluate a single rule against the graph.

        Args:
            rule: The integrity rule to evaluate

        Returns:
            List of violations (empty if rule is satisfied)
        """
        # Get all nodes of the target type
        nodes = self.graph.get_nodes_by_type(rule.applies_to_node_type)

        violations = []

        for node in nodes:
            violation = self._check_node_against_rule(node, rule)
            if violation is not None:
                violations.append(violation)

        return violations

    def evaluate_all_rules(self, rules: list[IntegrityRule]) -> list[RuleViolation]:
        """
        Evaluate multiple rules against the graph.

        Args:
            rules: List of integrity rules to evaluate

        Returns:
            List of all violations found
        """
        all_violations = []

        for rule in rules:
            violations = self.evaluate_rule(rule)
            all_violations.extend(violations)

        return all_violations

    def get_violations_by_severity(
        self, rules: list[IntegrityRule], severity: Severity
    ) -> list[RuleViolation]:
        """
        Get violations of a specific severity level.

        Args:
            rules: List of integrity rules to evaluate
            severity: Severity level to filter by

        Returns:
            List of violations matching the severity
        """
        all_violations = self.evaluate_all_rules(rules)
        return [v for v in all_violations if v.severity == severity]

    def get_violations_for_node(
        self, node_id: str, rules: list[IntegrityRule]
    ) -> list[RuleViolation]:
        """
        Get all violations for a specific node.

        Args:
            node_id: Node to check
            rules: List of integrity rules to evaluate

        Returns:
            List of violations for this node
        """
        all_violations = self.evaluate_all_rules(rules)
        return [v for v in all_violations if v.node_id == node_id]

    # ── Internal Evaluation Logic ────────────────────────────────────

    def _check_node_against_rule(
        self, node: BaseNode, rule: IntegrityRule
    ) -> RuleViolation | None:
        """
        Check if a single node violates a rule.

        Returns:
            RuleViolation if violated, None if satisfied
        """
        if rule.rule_type == "minimum_outgoing_edge_count":
            return self._check_minimum_outgoing(node, rule)

        elif rule.rule_type == "minimum_outgoing_edge_count_or_flag":
            return self._check_minimum_outgoing_or_flag(node, rule)

        elif rule.rule_type == "exact_outgoing_edge_count":
            return self._check_exact_outgoing(node, rule)

        elif rule.rule_type == "minimum_incoming_edge_count":
            return self._check_minimum_incoming(node, rule)

        else:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")

    def _check_minimum_outgoing(
        self, node: BaseNode, rule: IntegrityRule
    ) -> RuleViolation | None:
        """Check minimum outgoing edge count."""
        # Get ALL outgoing edges, not filtered by type
        edges = self.graph.get_outgoing_edges(node.id)

        # Filter by target node types only
        valid_edges = [
            e for e in edges if self._is_valid_target(e.target_id, rule.target_node_types)
        ]

        actual_count = len(valid_edges)
        expected_count = rule.minimum_count or 0

        if actual_count < expected_count:
            return RuleViolation(
                rule_id=rule.id,
                node_id=node.id,
                node_name=node.name,
                severity=rule.severity,
                message=rule.failure_message_template.format(subject_name=node.name),
                expected_count=expected_count,
                actual_count=actual_count,
            )

        return None

    def _check_minimum_outgoing_or_flag(
        self, node: BaseNode, rule: IntegrityRule
    ) -> RuleViolation | None:
        """Check minimum outgoing edge count, but allow explicit 'no edges' flag."""
        # Check if node has explicit flag (e.g., Component.has_no_dependencies)
        if rule.allow_explicit_none_flag:
            # Check for common flag patterns
            if hasattr(node, "has_no_dependencies") and node.has_no_dependencies:
                return None
            if hasattr(node, "has_no_edges") and node.has_no_edges:
                return None

        # Otherwise, check normally
        return self._check_minimum_outgoing(node, rule)

    def _check_exact_outgoing(self, node: BaseNode, rule: IntegrityRule) -> RuleViolation | None:
        """Check exact outgoing edge count."""
        # Get ALL outgoing edges, not filtered by type
        edges = self.graph.get_outgoing_edges(node.id)

        # Filter by target node types only
        valid_edges = [
            e for e in edges if self._is_valid_target(e.target_id, rule.target_node_types)
        ]

        actual_count = len(valid_edges)
        expected_count = rule.exact_count or 0

        if actual_count != expected_count:
            return RuleViolation(
                rule_id=rule.id,
                node_id=node.id,
                node_name=node.name,
                severity=rule.severity,
                message=rule.failure_message_template.format(subject_name=node.name),
                expected_count=expected_count,
                actual_count=actual_count,
            )

        return None

    def _check_minimum_incoming(
        self, node: BaseNode, rule: IntegrityRule
    ) -> RuleViolation | None:
        """Check minimum incoming edge count."""
        # Get ALL incoming edges, not filtered by type
        edges = self.graph.get_incoming_edges(node.id)

        # Filter by source node types only
        valid_edges = [
            e for e in edges if self._is_valid_source(e.source_id, rule.target_node_types)
        ]

        actual_count = len(valid_edges)
        expected_count = rule.minimum_count or 0

        if actual_count < expected_count:
            return RuleViolation(
                rule_id=rule.id,
                node_id=node.id,
                node_name=node.name,
                severity=rule.severity,
                message=rule.failure_message_template.format(subject_name=node.name),
                expected_count=expected_count,
                actual_count=actual_count,
            )

        return None

    def _is_valid_target(self, target_id: str, valid_types: list[str]) -> bool:
        """Check if target node is of a valid type."""
        target = self.graph.get_node(target_id)
        if target is None:
            return False

        target_type = self.graph._get_node_type(target)
        return target_type in valid_types

    def _is_valid_source(self, source_id: str, valid_types: list[str]) -> bool:
        """Check if source node is of a valid type."""
        source = self.graph.get_node(source_id)
        if source is None:
            return False

        source_type = self.graph._get_node_type(source)
        return source_type in valid_types
