"""
Domain configuration — everything a domain brings to the conversation engine.

A DomainConfig is a frozen value object that describes a complete project
configuration.  It is the single serialisation target for project-level
persistence: one ``DomainConfig`` ↔ one project in the store.

All fields are optional.  When a field is ``None`` the conversation graph
treats it as "not yet available" — a deterministic gather node can fill
the gap (ask the human, load from DB, etc.) without involving the LLM.

Usage::

    from conversation_engine.models.domain_config import DomainConfig

    cfg = DomainConfig(
        project_name="my-project",
        project_spec=spec,
        rules=rules,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# from conversation_engine.models.query_node import GraphQueryPattern
from conversation_engine.models.rule_node import IntegrityRule
from conversation_engine.models.validation_quiz import ValidationQuiz, FactualQuiz, ReasoningQuiz
from conversation_engine.models.project_spec import ProjectSpecification


@dataclass(frozen=True)
class ControlSet:
    """
    A subset of the domain DomainConfig specification for a project.

    This is a pure value object — no behavior, no side-effects.
    The ``ConversationContext`` implementation *consumes* a DomainConfig;
    graph nodes *inspect* it to decide what still needs gathering.

    Attributes:
        project_name: Unique human-readable project identifier.
        project_spec: Flat, business-level project specification (no graph).
        rules: Integrity rules the graph must satisfy.
        quiz: LLM pre-flight validation questions.
        query_patterns: Reusable graph query patterns for AI reasoning.
        system_prompt: The system prompt that teaches an LLM about this domain.
        metadata: Arbitrary extra data (version, owner, timestamps, etc.).
    """

    project_name: str
    rules: Optional[List[IntegrityRule]] = None
    quiz: Optional[List[ValidationQuiz]] = None
    # query_patterns: Optional[List[GraphQueryPattern]] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainConfig:
    """
    Complete domain specification for a project.

    This is a pure value object — no behavior, no side-effects.
    The ``ConversationContext`` implementation *consumes* a DomainConfig;
    graph nodes *inspect* it to decide what still needs gathering.

    Attributes:
        project_name: Unique human-readable project identifier.
        project_spec: Flat, business-level project specification (no graph).
        rules: Integrity rules the graph must satisfy.
        quiz: LLM pre-flight validation questions.
        query_patterns: Reusable graph query patterns for AI reasoning.
        system_prompt: The system prompt that teaches an LLM about this domain.
        metadata: Arbitrary extra data (version, owner, timestamps, etc.).
    """

    project_name: str
    project_spec: Optional[ProjectSpecification] = None
    rules: Optional[List[IntegrityRule]] = None
    quiz: Optional[List[ValidationQuiz]] = None
    # query_patterns: Optional[List[GraphQueryPattern]] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # # ── Convenience Methods ────────────────────────────────────────────

    def with_project_name(self, project_name: str) -> "DomainConfig":
        """Return a new DomainConfig with the project_name changed."""
        return DomainConfig(
            project_name=project_name,
            project_spec=self.project_spec,
            rules=self.rules,
            quiz=self.quiz,
            # query_patterns=self.query_patterns,
            system_prompt=self.system_prompt,
            metadata=self.metadata,
        )

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full domain config to a JSON-safe dict.

        This is the payload that a ``ProjectStore`` writes to disk or DB.
        """
        return {
            "project_name": self.project_name,
            "project_spec": (
                self.project_spec.model_dump() if self.project_spec is not None else None
            ),
            "rules": ([r.model_dump() for r in self.rules] if self.rules is not None else None),
            "quiz": ([_quiz_to_dict(q) for q in self.quiz] if self.quiz is not None else None),
            # "query_patterns": (
            #     [p.model_dump() for p in self.query_patterns]
            #     if self.query_patterns is not None
            #     else None
            # ),
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainConfig":
        """
        Reconstruct a DomainConfig from the dict produced by ``to_dict()``.
        """
        spec_data = data.get("project_spec")
        rules_data = data.get("rules")
        quiz_data = data.get("quiz")
        # patterns_data = data.get("query_patterns")

        return cls(
            project_name=data["project_name"],
            project_spec=(
                ProjectSpecification.model_validate(spec_data) if spec_data is not None else None
            ),
            rules=(
                [IntegrityRule.model_validate(r) for r in rules_data]
                if rules_data is not None
                else None
            ),
            quiz=([_quiz_from_dict(q) for q in quiz_data] if quiz_data is not None else None),
            # query_patterns=(
            #     [GraphQueryPattern.model_validate(p) for p in patterns_data]
            #     if patterns_data is not None
            #     else None
            # ),
            system_prompt=data.get("system_prompt"),
            metadata=data.get("metadata", {}),
        )


# ── ValidationQuiz dict helpers (dataclass, not Pydantic) ────────────


def _quiz_to_dict(q: ValidationQuiz) -> Dict[str, Any]:
    base_dict = {
        "id": q.id,
        "name": q.name,
        "question": q.question,
        "weight": q.weight,
        "min_score": q.min_score,
        "quiz_type": q.quiz_type,
    }

    if q.quiz_type == "factual":
        base_dict["expected_answer"] = q.expected_answer
    elif q.quiz_type == "reasoning":
        base_dict["evaluation_criteria"] = q.evaluation_criteria

    return base_dict


def _quiz_from_dict(d: Dict[str, Any]) -> ValidationQuiz:
    quiz_type = d.get("quiz_type", "factual")  # Default to factual for backward compatibility

    base_kwargs = {
        "id": d["id"],
        "name": d["name"],
        "question": d["question"],
        "weight": d.get("weight", 1.0),
        "min_score": d.get("min_score", 0.5),
    }

    if quiz_type == "factual":
        # Handle new format or backward compatibility
        if "expected_answer" in d:
            base_kwargs["expected_answer"] = d["expected_answer"]
        elif "required_concepts" in d:
            # Backward compatibility: convert list to string
            base_kwargs["expected_answer"] = ", ".join(d["required_concepts"])
        else:
            base_kwargs["expected_answer"] = ""
        return FactualQuiz(**base_kwargs)
    elif quiz_type == "reasoning":
        if "evaluation_criteria" in d:
            base_kwargs["evaluation_criteria"] = d["evaluation_criteria"]
        elif "required_concepts" in d:
            # Backward compatibility: convert list to string
            base_kwargs["evaluation_criteria"] = ", ".join(d["required_concepts"])
        else:
            base_kwargs["evaluation_criteria"] = ""
        return ReasoningQuiz(**base_kwargs)
    else:
        raise ValueError(f"Unknown quiz_type: {quiz_type}")
