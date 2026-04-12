"""
Integrity rule models for graph validation.

These models define machine-readable constraints that the knowledge graph must satisfy.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from conversation_engine.models.base import EdgeType, NodeType, BaseNode

RuleType = Literal[
    "minimum_outgoing_edge_count",
    "minimum_outgoing_edge_count_or_flag",
    "exact_outgoing_edge_count",
    "minimum_incoming_edge_count",
]

Severity = Literal["low", "medium", "high"]


class IntegrityRule(BaseNode):
    """
    An integrity rule defines a constraint on the knowledge graph structure.

    Rules are evaluated by the validation system to detect gaps or inconsistencies.
    """

    node_type: NodeType = Field(NodeType.RULE, description="Type of this node")
    description: str = Field(..., description="Explanation of what this rule enforces")

    applies_to_node_type: NodeType = Field(..., description="The node type this rule applies to")
    rule_type: RuleType = Field(..., description="Type of validation check")

    target_node_types: List[NodeType] = Field(
        ..., description="Valid target node types for the edge"
    )

    minimum_count: Optional[int] = Field(
        None, description="Minimum number of edges required (for minimum_* rules)"
    )
    exact_count: Optional[int] = Field(
        None, description="Exact number of edges required (for exact_* rules)"
    )
    allow_explicit_none_flag: bool = Field(
        False, description="If true, a node can explicitly declare it has no edges via a flag"
    )

    severity: Severity = Field(..., description="Severity level of violations")
    failure_message_template: str = Field(
        ..., description="Template for error messages (can include {subject_name} placeholder)"
    )

    model_config = {"frozen": True}
