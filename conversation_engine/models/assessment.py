"""
Assessment output models for AI findings.

These models define the structured output format for AI assessments
of the knowledge graph.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from conversation_engine.models.rule_node import Severity

AssessmentType = Literal[
    "missing_goal_coverage",
    "missing_requirement_realization",
    "missing_component_dependencies",
    "orphan_decision",
    "orphan_constraint",
    "undocumented_feature",
    "component_without_design",
    "use_case_without_scenarios",
]

Confidence = Literal["low", "medium", "high"]


class Assessment(BaseModel):
    """
    A structured finding from AI analysis of the knowledge graph.

    Assessments identify gaps, inconsistencies, or completeness issues.
    """

    id: str = Field(..., description="Unique identifier for this assessment")
    assessment_type: AssessmentType = Field(..., description="Type of issue detected")
    severity: Severity = Field(..., description="Severity level")

    subject_ids: list[str] = Field(..., description="IDs of nodes involved in this finding")
    finding: str = Field(..., description="Human-readable finding description")
    evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting this finding"
    )

    related_rule_ids: list[str] = Field(
        default_factory=list, description="IDs of integrity rules that were violated"
    )
    suggested_actions: list[str] = Field(
        default_factory=list, description="Suggested actions to address this finding"
    )

    confidence: Confidence = Field(..., description="AI confidence level in this assessment")

    conversation_turn_id: str | None = Field(
        None, description="ID of the conversation turn that addressed this assessment"
    )
    resolved: bool = Field(
        False, description="Whether this assessment has been resolved in conversation"
    )
