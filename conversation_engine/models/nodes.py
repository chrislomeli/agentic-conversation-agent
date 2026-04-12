"""
Node type models for the knowledge graph.

Each node type represents a specific kind of architectural knowledge.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import Field

from conversation_engine.models.base import BaseNode, NodeType


"""
    project_node = Project(
        id=_slugify("project", project.project_name),
        name=project.project_name,
        system_prompt=project.system_prompt or "",
        metadata=json.dumps(project.metadata),
    )
"""


class Project(BaseNode):
    """
    A top level Project node.
    """

    node_type: NodeType = Field(NodeType.PROJECT, description="Type of this node")
    system_prompt: Optional[str] = Field(None, description="top level project node")
    metadata: Optional[str] = Field(None, description="top level project node")


class Feature(BaseNode):
    """
    A feature represents a high-level system capability or product offering.
    """

    node_type: NodeType = Field(NodeType.FEATURE, description="Type of this node")
    description: str = Field(..., description="Detailed description of the feature")


class Goal(BaseNode):
    """
    A goal represents a desired outcome or objective.
    """

    node_type: NodeType = Field(NodeType.GOAL, description="Type of this node")
    statement: str = Field(..., description="Goal statement describing the desired outcome")


class GuidingPrinciple(BaseNode):
    """
    A guiding principle represents a design or architectural principle.
    """

    node_type: NodeType = Field(NodeType.GUIDING_PRINCIPLE, description="Type of this node")
    statement: str = Field(..., description="The principle statement")


RequirementType = Literal["functional", "non_functional", "constraint"]


class Requirement(BaseNode):
    """
    A requirement represents a specific system need or constraint.
    """

    node_type: NodeType = Field(NodeType.REQUIREMENT, description="Type of this node")
    requirement_type: Optional[RequirementType] = Field(None, description="Type of requirement")
    description: Optional[str] = Field(None, description="Detailed description of the requirement")


class Capability(BaseNode):
    """
    A capability represents an observable system behavior.
    """

    node_type: NodeType = Field(NodeType.CAPABILITY, description="Type of this node")
    description: Optional[str] = Field(
        None, description="Description of what the capability enables"
    )


class UseCase(BaseNode):
    """
    A use case represents a specific user interaction or workflow.
    """

    node_type: NodeType = Field(NodeType.USE_CASE, description="Type of this node")
    description: Optional[str] = Field(None, description="Description of the use case")


class Scenario(BaseNode):
    """
    A scenario represents a concrete instance or example of a use case.
    """

    node_type: NodeType = Field(NodeType.SCENARIO, description="Type of this node")
    description: Optional[str] = Field(None, description="Description of the scenario")


class DesignArtifact(BaseNode):
    """
    A design artifact represents a design decision or architectural element.
    """

    node_type: NodeType = Field(NodeType.DESIGN_ARTIFACT, description="Type of this node")
    statement: str = Field(..., description="Statement describing the design artifact")


class Decision(BaseNode):
    """
    A decision represents an architectural or design decision.
    """

    node_type: NodeType = Field(NodeType.DECISION, description="Type of this node")
    statement: str = Field(..., description="The decision that was made")
    rationale: Optional[str] = Field(None, description="Why this decision was made")


class Constraint(BaseNode):
    """
    A constraint represents a limitation or restriction on the system.
    """

    node_type: NodeType = Field(NodeType.CONSTRAINT, description="Type of this node")
    statement: str = Field(..., description="The constraint statement")


class Component(BaseNode):
    """
    A component represents a system module or architectural component.
    """

    node_type: NodeType = Field(NodeType.COMPONENT, description="Type of this node")
    description: Optional[str] = Field(None, description="Description of the component's purpose")
    has_no_dependencies: bool = Field(
        False, description="Explicitly marks that this component has no dependencies"
    )


StepStatus = Literal["pending", "in_progress", "done", "blocked"]


class Step(BaseNode):
    """
    A step represents a concrete work item that realises a requirement.

    Steps track progress via status and percentage, and may reference
    other steps as blockers.
    """

    node_type: NodeType = Field(NodeType.STEP, description="Type of this node")
    description: Optional[str] = Field(
        None, description="Description of what this step accomplishes"
    )
    status: StepStatus = Field("pending", description="Current status of this step")
    percentage: int = Field(
        0,
        description="Completion percentage (0-100)",
        ge=0,
        le=100,
    )
    blocker_refs: List[str] = Field(
        default_factory=list, description="Names of other steps that block this step"
    )
    has_no_dependencies: bool = Field(
        False, description="Explicitly marks that this step has no external dependencies"
    )


class Dependency(BaseNode):
    """
    A dependency represents an external system, library, or service.
    """

    node_type: NodeType = Field(NodeType.DEPENDENCY, description="Type of this node")
    description: Optional[str] = Field(None, description="Description of the dependency")


class DocumentationArtifact(BaseNode):
    """
    A documentation artifact represents documentation or explanatory content.
    """

    node_type: NodeType = Field(NodeType.DOCUMENTATION_ARTIFACT, description="Type of this node")
    description: Optional[str] = Field(None, description="Description of the documentation")
