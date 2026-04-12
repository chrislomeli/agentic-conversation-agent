"""
llm — Backwards-compatibility shim.

Reusable LLM protocols and adapters have moved to ``commons.llm``.
Domain-specific ValidationQuiz stays in conversation_engine.models.
"""

from commons.llm import (
    CallLLM,
    LLMRequest,
    LLMResponse,
    LLMValidator,
    LLMValidatorReport,
    OpenAICallLLM,
    QuizResult,
    call_llm_stub,
    make_openai_llm,
    quiz_report_summary,
)

from conversation_engine.models.validation_quiz import ValidationQuiz

__all__ = [
    "CallLLM",
    "LLMRequest",
    "LLMResponse",
    "call_llm_stub",
    "make_openai_llm",
    "OpenAICallLLM",
    "LLMValidator",
    "LLMValidatorReport",
    "ValidationQuiz",
    "QuizResult",
    "quiz_report_summary",
]
