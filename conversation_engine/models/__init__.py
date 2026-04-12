"""
Domain models for the Reusable Conversation Engine.
"""

# from conversation_engine.models.query_node import (
#     QueryIntent,
#     OutputKind,
#     EdgeCheck,
#     TraversalSpec,
#     PathStep,
#     GraphQueryPattern,
# )
from conversation_engine.models.assessment import (
    Assessment,
    AssessmentType,
)
from conversation_engine.models.base import BaseEdge, BaseNode, EdgeType, NodeType
from conversation_engine.models.nodes import (
    Constraint,
    Decision,
    Dependency,
    DesignArtifact,
    DocumentationArtifact,
    Feature,
    Goal,
    GuidingPrinciple,
    Requirement,
    Scenario,
    Step,
    StepStatus,
    UseCase,
)
from conversation_engine.models.rule_node import (
    IntegrityRule,
    RuleType,
    Severity,
)
from conversation_engine.models.traceability import (
    GoalRequirementTrace,
    RequirementStepTrace,
    StepDependencyTrace,
)

# DomainConfig is intentionally NOT imported here to avoid a circular
# import (domain_config → storage.graph → models.base → models/__init__).
# Import it directly:  from conversation_engine.models.domain_config import DomainConfig

__all__ = [
    "NodeType",
    "EdgeType",
    "BaseNode",
    "BaseEdge",
    "Feature",
    "Goal",
    "GuidingPrinciple",
    "Requirement",
    "Step",
    "StepStatus",
    "UseCase",
    "Scenario",
    "DesignArtifact",
    "Decision",
    "Constraint",
    "Dependency",
    "DocumentationArtifact",
    "GoalRequirementTrace",
    "RequirementStepTrace",
    "StepDependencyTrace",
    "RuleType",
    "Severity",
    # "IntegrityRule",
    # "QueryIntent",
    # "OutputKind",
    # "EdgeCheck",
    # "TraversalSpec",
    # "PathStep",
    # "GraphQueryPattern",
    "AssessmentType",
    "Assessment",
]
