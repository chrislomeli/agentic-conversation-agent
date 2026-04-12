"""
QuizQuestion — Domain-agnostic protocol for LLM pre-run validation questions.

This replaces the domain-specific ValidationQuiz (which inherits BaseNode)
with a simple protocol that any domain can satisfy without importing
domain models.

Any object with these attributes works:
  - question: str
  - quiz_type: "factual" or "reasoning"
  - expected_answer: str (for factual) or evaluation_criteria: str (for reasoning)
  - weight: float
  - min_score: float
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class QuizQuestion(Protocol):
    """
    Protocol for a single LLM validation quiz question.

    Domain implementations can use any class (dataclass, Pydantic model,
    etc.) as long as it has these attributes.
    """

    question: str
    quiz_type: str  # "factual" or "reasoning"
    weight: float
    min_score: float


@dataclass(frozen=True)
class FactualQuizQuestion:
    """Simple factual quiz question — checks for keyword presence."""

    question: str
    expected_answer: str
    quiz_type: Literal["factual"] = "factual"
    weight: float = 1.0
    min_score: float = 0.5


@dataclass(frozen=True)
class ReasoningQuizQuestion:
    """Reasoning quiz question — checks for evaluation criteria presence."""

    question: str
    evaluation_criteria: str
    quiz_type: Literal["reasoning"] = "reasoning"
    weight: float = 1.0
    min_score: float = 0.5
