"""
Backwards-compatible re-export.

The canonical location is now ``conversation_engine.models.project_spec``.
This shim ensures existing imports keep working.
"""

from conversation_engine.models.project_spec import (  # noqa: F401
    ConstraintSpec,
    DependencySpec,
    GoalSpec,
    ProjectSnapshot,  # alias for ProjectSpecification
    ProjectSpecification,
    RequirementSpec,
    StepSpec,
)
