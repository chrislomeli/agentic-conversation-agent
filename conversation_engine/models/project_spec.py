"""
ProjectSpecification — flat, business-level project description.

This is the payload the LLM produces and consumes.  It uses human-readable
names instead of IDs and inline references instead of edges.  A facade
converts between ``ProjectSpecification`` and ``KnowledgeGraph`` at runtime
when graph operations (validation, queries) are needed.

The LLM never sees nodes, edges, or IDs.

Design principles:
- Every spec model uses **name-based references** (e.g. ``goal_ref``)
- The facade resolves names → IDs and wires the correct edge types/directions
- Adding a new concept = add a spec here + add conversion in the facade
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Individual spec models ─────────────────────────────────────────


class GoalSpec(BaseModel):
    """A desired outcome or objective."""

    name: str = Field(..., description="Human-readable goal name")
    statement: str = Field(..., description="Goal statement describing the desired outcome")


class RequirementSpec(BaseModel):
    """A specific system need linked to a goal."""

    name: str = Field(..., description="Human-readable requirement name")
    goal_ref: str = Field(..., description="Name of the goal this requirement satisfies")
    requirement_type: Literal["functional", "non_functional", "constraint"] | None = Field(
        None, description="Type of requirement"
    )
    description: str | None = Field(None, description="Detailed description")


StepStatus = Literal["pending", "in_progress", "done", "blocked"]


class StepSpec(BaseModel):
    """A concrete work item that realises a requirement."""

    name: str = Field(..., description="Human-readable step name")
    requirement_refs: list[str] = Field(
        default_factory=list,
        description="Names of requirements this step realises",
    )
    dependency_refs: list[str] = Field(
        default_factory=list,
        description="Names of dependencies this step depends on",
    )
    blocker_refs: list[str] = Field(
        default_factory=list,
        description="Names of other steps that block this step",
    )
    status: StepStatus = Field(
        "pending",
        description="Current status: pending, in_progress, done, or blocked",
    )
    percentage: int = Field(
        0,
        description="Completion percentage (0-100)",
        ge=0,
        le=100,
    )
    has_no_dependencies: bool = Field(
        False,
        description="Explicitly marks that this step has no external dependencies",
    )
    description: str | None = Field(
        None, description="Description of what this step accomplishes"
    )


# ── Backwards-compatible alias ────────────────────────────────────
ComponentSpec = StepSpec


class ConstraintSpec(BaseModel):
    """A limitation or restriction on the system."""

    name: str = Field(..., description="Human-readable constraint name")
    statement: str = Field(..., description="The constraint statement")


class DependencySpec(BaseModel):
    """An external system, library, or service."""

    name: str = Field(..., description="Human-readable dependency name")
    description: str | None = Field(None, description="Description of the dependency")


# ── Top-level specification ───────────────────────────────────────


class ProjectSpecification(BaseModel):
    """
    Flat, business-level representation of a project.

    This is the only structure the LLM interacts with.  It contains no
    node IDs, no edge types, and no graph-level concepts.  A facade
    converts it to/from the internal ``KnowledgeGraph`` when needed.
    """

    project_name: str = Field(..., description="Unique project identifier")
    description: str | None = Field(
        None,
        description="Brief description of what this project is and what problem it solves",
    )
    goals: list[GoalSpec] = Field(default_factory=list, description="Project goals")
    requirements: list[RequirementSpec] = Field(
        default_factory=list, description="Requirements linked to goals"
    )
    steps: list[StepSpec] = Field(
        default_factory=list, description="Steps (work items) linked to requirements"
    )
    constraints: list[ConstraintSpec] = Field(
        default_factory=list, description="System constraints"
    )
    dependencies: list[DependencySpec] = Field(
        default_factory=list, description="External dependencies"
    )

    model_config = {"frozen": False}


# ── Backwards-compatible alias ────────────────────────────────────
ProjectSnapshot = ProjectSpecification
