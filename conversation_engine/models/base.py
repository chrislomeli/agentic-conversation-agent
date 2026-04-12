"""
Base models for nodes and edges in the knowledge graph.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Enum for all node types in the knowledge graph."""

    PROJECT = "project"
    FEATURE = "feature"
    GOAL = "goal"
    GUIDING_PRINCIPLE = "guiding_principle"
    REQUIREMENT = "requirement"
    CAPABILITY = "capability"
    USE_CASE = "use_case"
    SCENARIO = "scenario"
    DESIGN_ARTIFACT = "design_artifact"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    COMPONENT = "component"
    STEP = "step"
    DEPENDENCY = "dependency"
    DOCUMENTATION_ARTIFACT = "documentation_artifact"
    RULE = "rule"
    QUIZ = "quiz"
    QUERY_PATTERN = "query_pattern"


EdgeType = Literal[
    "HAS_GOAL",
    "HAS_REQUIREMENT",
    "HAS_COMPONENT",
    "HAS_STEP",
    "HAS_DEPENDENCY",
    "HAS_CONSTRAINT",
    "HAS_RULE",
    "HAS_REASONING_QUIZ",
    "HAS_FACTUAL_QUIZ",
    "HAS_QUIZ",
    "HAS_QUERY_PATTERN",
    "SATISFIED_BY",
    "REALIZED_BY",
    "DEPENDS_ON",
    "BLOCKED_BY",
    "CONSTRAINS",
    "DESCRIBED_BY",
    "DOCUMENTED_BY",
    "SUPPORTS",
    "INFORMS",
    "HAS_SCENARIO",
    "INSTANCE_OF",
]


class BaseNode(BaseModel):
    """
    Base class for all knowledge graph nodes.

    All nodes have an ID, name, and type. Subclasses add domain-specific fields.
    IDs are immutable; content fields can be updated.
    """

    node_type: NodeType = Field(..., description="Type of this node")
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this node"
    )
    name: str = Field(default_factory=lambda: "missing_name", description="Human-readable name")

    model_config = {"frozen": False}


class BaseEdge(BaseModel):
    """
    Base class for all knowledge graph edges.

    Edges represent typed relationships between nodes.
    """

    edge_type: EdgeType = Field(..., description="Type of relationship")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")

    model_config = {"frozen": True}
