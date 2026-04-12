"""
Validation quiz model — a pure data definition for LLM pre-run validation.

This lives in the models layer because it is a domain data structure with
zero infrastructure dependencies.  The infrastructure layer
(``infrastructure.llm.validator``) *consumes* it at runtime.
"""

from __future__ import annotations

from pydantic import Field
from typing import Literal, Annotated, Union


from conversation_engine.models import BaseNode, NodeType
from enum import Enum


class QuizType(str, Enum):
    REASONING = "reasoning"  # evaluated by LLM judge using rubric
    FACTUAL = "factual"  # evaluated by exact or near-exact match


class ReasoningQuiz(BaseNode):
    node_type: NodeType = Field(NodeType.QUIZ, description="Type of this node")
    quiz_type: Literal[QuizType.REASONING] = QuizType.REASONING
    question: str
    evaluation_criteria: str  # required, not Optional
    weight: float = 1.0
    min_score: float = 0.5


class FactualQuiz(BaseNode):
    node_type: NodeType = Field(NodeType.QUIZ, description="Type of this node")
    quiz_type: Literal[QuizType.FACTUAL] = QuizType.FACTUAL
    question: str
    expected_answer: str  # required, not Optional
    weight: float = 1.0
    min_score: float = 0.5


ValidationQuiz = Annotated[Union[ReasoningQuiz, FactualQuiz], Field(discriminator="quiz_type")]
