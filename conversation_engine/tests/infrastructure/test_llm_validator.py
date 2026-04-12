"""
Tests for LLM pre-run validator.

Covers:
- Scoring logic: required concepts, prohibited concepts, edge cases
- LLMValidator: smart LLM passes, dumb LLM fails, failed LLM call
- ValidationQuiz: weight, min_score
- quiz_report_summary formatting
- Architectural quiz with a mock LLM that echoes the system prompt
- Architectural quiz with a clueless LLM that fails
"""


from conversation_engine.infrastructure.llm.architectural_quiz import (
    ARCHITECTURAL_QUIZ,
    ARCHITECTURAL_SYSTEM_PROMPT,
)
from conversation_engine.infrastructure.llm.protocols import (
    LLMRequest,
    LLMResponse,
)
from conversation_engine.infrastructure.llm.validator import (
    FactualQuiz,
    LLMValidator,
    LLMValidatorReport,
    QuizResult,
    _score_response,
    quiz_report_summary,
)

# ── Scoring tests ──────────────────────────────────────────────────


class TestScoreResponse:
    def test_all_concepts_found(self):
        quiz = FactualQuiz(
            question="What colors?",
            expected_answer="red, blue, green",
        )
        result = _score_response("I see red, blue, and green.", quiz)
        assert result.score == 1.0
        assert result.passed
        assert result.missing_concepts == []

    def test_partial_concepts(self):
        quiz = FactualQuiz(
            question="What colors?",
            expected_answer="red, blue, green, yellow",
            min_score=0.5,
        )
        result = _score_response("I see red and blue.", quiz)
        assert result.score == 0.5
        assert result.passed  # 0.5 >= 0.5
        assert set(result.missing_concepts) == {"green", "yellow"}

    def test_no_concepts_found(self):
        quiz = FactualQuiz(
            question="What colors?",
            expected_answer="red, blue",
        )
        result = _score_response("I don't know anything about this.", quiz)
        assert result.score == 0.0
        assert not result.passed

    # NOTE: prohibited_concepts feature removed in new quiz structure
    # def test_prohibited_concepts_penalize(self):
    #     quiz = ValidationQuiz(
    #         question="What is your role?",
    #         evaluation_criteria=["advise", "suggest"],
    #         prohibited_concepts=["i modify the graph"],
    #     )
    #     result = _score_response(
    #         "I advise and suggest changes. I modify the graph directly.",
    #         quiz,
    #     )
    #     # Found 2, prohibited 1 → (2-1)/2 = 0.5
    #     assert result.score == 0.5
    #     assert "i modify the graph" in result.prohibited_found

    # NOTE: prohibited_concepts feature removed in new quiz structure
    # def test_prohibited_only(self):
    #     quiz = ValidationQuiz(
    #         question="Tell me about yourself.",
    #         evaluation_criteria=[],
    #         prohibited_concepts=["i am sentient"],
    #     )
    #     # No prohibited found → perfect
    #     result = _score_response("I am an AI assistant.", quiz)
    #     assert result.score == 1.0
    #     assert result.passed

    # NOTE: prohibited_concepts feature removed in new quiz structure
    # def test_prohibited_found_no_required(self):
    #     quiz = ValidationQuiz(
    #         question="Tell me about yourself.",
    #         evaluation_criteria=[],
    #         prohibited_concepts=["i am sentient"],
    #     )
    #     result = _score_response("I am sentient and self-aware.", quiz)
    #     assert result.score == 0.0
    #     assert not result.passed

    def test_case_insensitive(self):
        quiz = FactualQuiz(
            question="What types?",
            expected_answer="Goal, Requirement",
        )
        result = _score_response("We have GOAL and requirement types.", quiz)
        assert result.score == 1.0

    def test_word_boundary_matching(self):
        """'goal' should not match 'goals' at word boundary... but actually
        regex \\bgoal\\b won't match 'goals'. Let's verify."""
        quiz = FactualQuiz(
            question="What?",
            expected_answer="goal",
        )
        # "goals" should NOT match "goal" with word boundaries
        result = _score_response("We have many goals here.", quiz)
        # Actually "goals" does NOT contain \bgoal\b because 's' follows
        # Wait - "goals" contains "goal" at a word boundary start but 's' is
        # a word char so \bgoal\b won't match inside "goals"
        # This is correct behavior - we want exact concept matching
        assert result.score == 0.0

    def test_min_score_threshold(self):
        quiz = FactualQuiz(
            question="What?",
            expected_answer="a, b, c, d",
            min_score=0.75,
        )
        # 2 out of 4 = 0.5 < 0.75 → fail
        result = _score_response("a and b are here.", quiz)
        assert not result.passed
        assert result.score == 0.5


# ── Mock LLMs ──────────────────────────────────────────────────────


def _smart_llm(request: LLMRequest) -> LLMResponse:
    """An LLM that echoes the system prompt back — guaranteed to contain all concepts."""
    return LLMResponse(
        content=request.system_prompt,
        model="smart-echo",
        success=True,
    )


def _dumb_llm(request: LLMRequest) -> LLMResponse:
    """An LLM that always responds with irrelevant nonsense."""
    return LLMResponse(
        content="I like pizza and sunshine. The weather is nice today.",
        model="dumb-stub",
        success=True,
    )


def _failing_llm(request: LLMRequest) -> LLMResponse:
    """An LLM whose calls always fail."""
    return LLMResponse(
        content="",
        model="failing",
        success=False,
        error="API connection refused",
    )


def _partial_llm(request: LLMRequest) -> LLMResponse:
    """An LLM that knows about goals and requirements but nothing else."""
    return LLMResponse(
        content=(
            "The system has goal and requirement node types connected by "
            "SATISFIED_BY edges. When a goal has no requirement, it produces "
            "a finding with a severity. I advise the user on what to fix. "
            "High severity issues should be addressed first, then medium, then low. "
            "When all checks pass, the architecture is complete."
        ),
        model="partial",
        success=True,
    )


# ── Validator tests ────────────────────────────────────────────────


class TestLLMValidator:
    def _simple_quiz(self) -> list[FactualQuiz]:
        return [
            FactualQuiz(
                question="What are node types?",
                expected_answer="goal, requirement",
                weight=1.0,
            ),
            FactualQuiz(
                question="What is your role?",
                expected_answer="advise",
                weight=1.0,
            ),
        ]

    def test_smart_llm_passes(self):
        validator = LLMValidator(
            llm=_smart_llm,
            system_prompt="I advise about goal and requirement types.",
            quiz=self._simple_quiz(),
            pass_threshold=0.7,
        )
        report = validator.run()

        assert report.passed
        assert report.weighted_score >= 0.7
        assert len(report.results) == 2
        assert all(r.passed for r in report.results)

    def test_dumb_llm_fails(self):
        validator = LLMValidator(
            llm=_dumb_llm,
            system_prompt="irrelevant",
            quiz=self._simple_quiz(),
            pass_threshold=0.7,
        )
        report = validator.run()

        assert not report.passed
        assert report.weighted_score < 0.7

    def test_failing_llm_gets_zero(self):
        validator = LLMValidator(
            llm=_failing_llm,
            system_prompt="anything",
            quiz=self._simple_quiz(),
            pass_threshold=0.1,
        )
        report = validator.run()

        assert not report.passed
        assert report.weighted_score == 0.0
        for r in report.results:
            assert r.score == 0.0
            assert not r.passed

    def test_weighted_scoring(self):
        quiz = [
            FactualQuiz(
                question="Easy question",
                expected_answer="pizza",
                weight=1.0,
            ),
            FactualQuiz(
                question="Hard question",
                expected_answer="quantum, entanglement",
                weight=3.0,  # 3x weight
            ),
        ]
        validator = LLMValidator(
            llm=_dumb_llm,  # only knows about pizza
            system_prompt="",
            quiz=quiz,
            pass_threshold=0.5,
        )
        report = validator.run()

        # Pizza found (1.0 * 1.0) + quantum/entanglement missed (0.0 * 3.0)
        # = 1.0 / 4.0 = 0.25
        assert report.weighted_score == 0.25
        assert not report.passed

    def test_threshold_edge(self):
        quiz = [
            FactualQuiz(
                question="Q1",
                expected_answer="pizza",
                weight=1.0,
            ),
        ]
        # Dumb LLM says "pizza" → score 1.0
        validator = LLMValidator(
            llm=_dumb_llm,
            system_prompt="",
            quiz=quiz,
            pass_threshold=1.0,
        )
        report = validator.run()
        assert report.passed
        assert report.weighted_score == 1.0

    def test_responses_captured(self):
        validator = LLMValidator(
            llm=_smart_llm,
            system_prompt="Hello world",
            quiz=[FactualQuiz(question="Q1", expected_answer="hello")],
        )
        report = validator.run()
        assert len(report.llm_responses) == 1
        assert report.llm_responses[0].model == "smart-echo"

    def test_empty_quiz(self):
        validator = LLMValidator(
            llm=_smart_llm,
            system_prompt="",
            quiz=[],
            pass_threshold=0.7,
        )
        report = validator.run()
        assert report.weighted_score == 0.0
        assert not report.passed


# ── Architectural quiz tests ───────────────────────────────────────


class TestArchitecturalQuiz:
    def test_smart_llm_passes_architectural_quiz(self):
        """An LLM that echoes the system prompt should pass the arch quiz."""
        validator = LLMValidator(
            llm=_smart_llm,
            system_prompt=ARCHITECTURAL_SYSTEM_PROMPT,
            quiz=ARCHITECTURAL_QUIZ,
            pass_threshold=0.7,
        )
        report = validator.run()

        assert report.passed, (
            f"Smart LLM failed architectural quiz with score "
            f"{report.weighted_score:.1%}:\n{quiz_report_summary(report)}"
        )

    def test_dumb_llm_fails_architectural_quiz(self):
        """A clueless LLM should fail the architectural quiz."""
        validator = LLMValidator(
            llm=_dumb_llm,
            system_prompt=ARCHITECTURAL_SYSTEM_PROMPT,
            quiz=ARCHITECTURAL_QUIZ,
            pass_threshold=0.7,
        )
        report = validator.run()
        assert not report.passed

    def test_partial_llm_score_range(self):
        """A partially-knowledgeable LLM should score somewhere in the middle."""
        validator = LLMValidator(
            llm=_partial_llm,
            system_prompt=ARCHITECTURAL_SYSTEM_PROMPT,
            quiz=ARCHITECTURAL_QUIZ,
            pass_threshold=0.7,
        )
        report = validator.run()

        # Should get some questions right but not all
        assert report.weighted_score > 0.0
        assert report.weighted_score < 1.0

    def test_quiz_has_reasonable_coverage(self):
        """The architectural quiz should cover key domain concepts."""
        all_required = set()
        for q in ARCHITECTURAL_QUIZ:
            # For FactualQuiz, check expected_answer
            if hasattr(q, "expected_answer"):
                concepts = [c.strip().lower() for c in q.expected_answer.split(",")]
                all_required.update(concepts)
            # For ReasoningQuiz, check evaluation_criteria (if any exist)
            elif hasattr(q, "evaluation_criteria"):
                concepts = [c.strip().lower() for c in q.evaluation_criteria.split(",")]
                all_required.update(concepts)

        # Must test for core node types
        assert "goal" in all_required
        assert "requirement" in all_required
        assert "step" in all_required

        # Must test for edge understanding
        assert "satisfied_by" in all_required

        # Must test for findings understanding
        assert "severity" in all_required
        assert "finding" in all_required or "message" in all_required

    # NOTE: prohibited_concepts feature removed in new quiz structure
    # def test_quiz_has_prohibited_concepts(self):
    #     """At least one question should check for prohibited behavior."""
    #     has_prohibited = any(q.prohibited_concepts for q in ARCHITECTURAL_QUIZ)
    #     assert has_prohibited, "Quiz should include hallucination detection"


# ── Report formatting tests ────────────────────────────────────────


class TestQuizReportSummary:
    def test_passing_report(self):
        report = LLMValidatorReport(
            results=[
                QuizResult(
                    question="Q1",
                    response="good answer",
                    found_concepts=["a", "b"],
                    missing_concepts=[],
                    prohibited_found=[],
                    score=1.0,
                    passed=True,
                    weight=1.0,
                ),
            ],
            weighted_score=1.0,
            passed=True,
            pass_threshold=0.7,
        )
        summary = quiz_report_summary(report)
        assert "PASSED" in summary
        assert "100.0%" in summary

    def test_failing_report(self):
        report = LLMValidatorReport(
            results=[
                QuizResult(
                    question="Q1",
                    response="bad answer",
                    found_concepts=[],
                    missing_concepts=["a", "b"],
                    prohibited_found=["bad_thing"],
                    score=0.0,
                    passed=False,
                    weight=1.0,
                ),
            ],
            weighted_score=0.0,
            passed=False,
            pass_threshold=0.7,
        )
        summary = quiz_report_summary(report)
        assert "FAILED" in summary
        assert "Prohibited" in summary
