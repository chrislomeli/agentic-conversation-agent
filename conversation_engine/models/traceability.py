"""
Traceability models for mapping relationships between layers.

These models capture the explicit traceability chains:
  Goal → Requirement → Step → Dependency
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class GoalRequirementTrace(BaseModel):
    """
    Maps a goal to the requirements that satisfy it.
    """

    goal_id: str = Field(..., description="Goal identifier")
    requirement_ids: List[str] = Field(
        default_factory=list, description="List of requirement IDs that satisfy this goal"
    )


class RequirementStepTrace(BaseModel):
    """
    Maps a requirement to the steps that realize it.
    """

    requirement_id: str = Field(..., description="Requirement identifier")
    step_ids: List[str] = Field(
        default_factory=list, description="List of step IDs that realize this requirement"
    )


class StepDependencyTrace(BaseModel):
    """
    Maps a step to its dependencies.
    """

    step_id: str = Field(..., description="Step identifier")
    dependency_ids: List[str] = Field(
        default_factory=list,
        description="List of dependency IDs (external systems, libraries, etc.)",
    )


# ── Backwards-compatible aliases ─────────────────────────────────
RequirementComponentTrace = RequirementStepTrace
ComponentDependencyTrace = StepDependencyTrace
