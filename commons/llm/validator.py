"""
LLM Pre-Run Validator — Quiz a fresh LLM before trusting it.

Before "hiring" an LLM for the conversation loop, we:
  1. Send it the same system prompt the loop will use
  2. Ask a battery of quiz questions about the domain
  3. Score each response by checking for required concepts
  4. Produce a pass/fail report with per-question breakdown

The validator is domain-agnostic: the quiz questions and system prompt
are injected.  Different domains supply different quizzes.

Usage:
    from commons.llm import CallLLM, LLMValidator, FactualQuizQuestion

    quiz = [
        FactualQuizQuestion(
            question="What node types exist in the knowledge graph?",
            expected_answer="goal, requirement, step",
        ),
    ]
    validator = LLMValidator(llm=my_llm, system_prompt=SYSTEM_PROMPT, quiz=quiz)
    report = validator.run()
    if not report.passed:
        print(quiz_report_summary(report))
        raise RuntimeError("LLM failed pre-run validation")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from commons.llm.protocols import (
    CallLLM,
    LLMRequest,
    LLMResponse,
)
from commons.llm.quiz import QuizQuestion


# ── Per-question result ─────────────────────────────────────────────


@dataclass
class QuizResult:
    """Result of scoring a single quiz question."""

    question: str
    response: str
    found_concepts: List[str]
    missing_concepts: List[str]
    prohibited_found: List[str]
    score: float  # 0.0–1.0
    passed: bool
    weight: float


# ── Overall report ──────────────────────────────────────────────────


@dataclass
class LLMValidatorReport:
    """Full report from a pre-run validation run."""

    results: List[QuizResult]
    weighted_score: float  # 0.0–1.0
    passed: bool
    pass_threshold: float
    llm_responses: List[LLMResponse] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Scoring logic ───────────────────────────────────────────────────


def _get_required_concepts(quiz: QuizQuestion) -> List[str]:
    """Extract required concepts from a quiz question, regardless of type."""
    if quiz.quiz_type == "factual":
        raw = getattr(quiz, "expected_answer", "")
    elif quiz.quiz_type == "reasoning":
        raw = getattr(quiz, "evaluation_criteria", "")
    else:
        raw = ""
    return [c.strip() for c in raw.split(",") if c.strip()]


def _score_response(
    response_text: str,
    quiz: QuizQuestion,
) -> QuizResult:
    """
    Score a single LLM response against its quiz question.

    Scoring:
      - Each required concept found: +1 point
      - Final score = found / total_required
    """
    text_lower = response_text.lower()
    required_concepts = _get_required_concepts(quiz)

    found = []
    missing = []
    for concept in required_concepts:
        # Use word-boundary-aware search for short concepts
        pattern = re.compile(r"\b" + re.escape(concept.lower()) + r"\b")
        if pattern.search(text_lower):
            found.append(concept)
        else:
            missing.append(concept)

    # Calculate score
    if not required_concepts:
        raw_score = 1.0
    else:
        raw_score = max(0.0, len(found) / len(required_concepts))

    score = min(1.0, raw_score)
    passed = score >= quiz.min_score

    return QuizResult(
        question=quiz.question,
        response=response_text,
        found_concepts=found,
        missing_concepts=missing,
        prohibited_found=[],
        score=score,
        passed=passed,
        weight=quiz.weight,
    )


# ── Validator ───────────────────────────────────────────────────────


class LLMValidator:
    """
    Pre-run validator that quizzes an LLM to verify domain understanding.

    Parameters
    ----------
    llm : CallLLM
        The LLM callable to validate.
    system_prompt : str
        The system prompt the LLM will receive (same one the loop uses).
    quiz : Sequence[QuizQuestion]
        Battery of questions to ask. Any object satisfying the QuizQuestion
        protocol works (FactualQuizQuestion, ReasoningQuizQuestion, or
        domain-specific subtypes like ValidationQuiz).
    pass_threshold : float
        Minimum weighted score to pass (0.0–1.0). Default 0.7.
    """

    def __init__(
        self,
        llm: CallLLM,
        system_prompt: str,
        quiz: Sequence[QuizQuestion],
        pass_threshold: float = 0.7,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._quiz = quiz
        self._pass_threshold = pass_threshold

    def run(self) -> LLMValidatorReport:
        """
        Run all quiz questions through the LLM and produce a report.

        Each question is sent as a separate LLM call with the same
        system prompt, simulating how the loop would interact.
        """
        results: List[QuizResult] = []
        responses: List[LLMResponse] = []

        for q in self._quiz:
            request = LLMRequest(
                system_prompt=self._system_prompt,
                user_message=q.question,
            )

            response = self._llm(request)
            responses.append(response)

            if not response.success:
                # LLM call itself failed — automatic zero
                concepts = _get_required_concepts(q)
                results.append(
                    QuizResult(
                        question=q.question,
                        response=response.error or "(LLM call failed)",
                        found_concepts=[],
                        missing_concepts=concepts,
                        prohibited_found=[],
                        score=0.0,
                        passed=False,
                        weight=q.weight,
                    )
                )
                continue

            result = _score_response(response.content, q)
            results.append(result)

        # Weighted score
        total_weight = sum(r.weight for r in results)
        if total_weight > 0:
            weighted_score = sum(r.score * r.weight for r in results) / total_weight
        else:
            weighted_score = 0.0

        passed = weighted_score >= self._pass_threshold

        return LLMValidatorReport(
            results=results,
            weighted_score=round(weighted_score, 4),
            passed=passed,
            pass_threshold=self._pass_threshold,
            llm_responses=responses,
        )


# ── Report formatting ───────────────────────────────────────────────


def quiz_report_summary(report: LLMValidatorReport) -> str:
    """
    Produce a human-readable summary of a validation report.
    """
    status = "PASSED" if report.passed else "FAILED"
    lines = [
        f"LLM Pre-Run Validation: {status}",
        f"Weighted Score: {report.weighted_score:.1%} (threshold: {report.pass_threshold:.1%})",
        "",
    ]

    for i, r in enumerate(report.results, 1):
        q_status = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{q_status}] Q{i}: {r.question}")
        lines.append(
            f"    Score: {r.score:.1%}  |  Found: {r.found_concepts}  |  Missing: {r.missing_concepts}"
        )
        if r.prohibited_found:
            lines.append(f"    WARNING: Prohibited concepts found: {r.prohibited_found}")

    return "\n".join(lines)
