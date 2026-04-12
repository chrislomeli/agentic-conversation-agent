"""
llm — Protocol-based LLM interaction layer.

Protocols
---------
CallLLM     Send a prompt with context to an LLM, get structured response back.

Implementations
---------------
call_llm_stub       Deterministic stub (no LLM dependency).
OpenAICallLLM       Real OpenAI implementation (requires API key).

Validation
----------
LLMValidator        Pre-run quiz to verify LLM domain understanding.
QuizQuestion        Domain-agnostic quiz question protocol.

Swap in any callable that matches the Protocol signature.
"""

from commons.llm.openai_adapter import (
    OpenAICallLLM,
    make_openai_llm,
)
from commons.llm.protocols import (
    CallLLM,
    LLMRequest,
    LLMResponse,
)
from commons.llm.quiz import (
    FactualQuizQuestion,
    QuizQuestion,
    ReasoningQuizQuestion,
)
from commons.llm.stub import call_llm_stub
from commons.llm.validator import (
    LLMValidator,
    LLMValidatorReport,
    QuizResult,
    quiz_report_summary,
)

__all__ = [
    "CallLLM",
    "LLMRequest",
    "LLMResponse",
    "call_llm_stub",
    "make_openai_llm",
    "OpenAICallLLM",
    "QuizQuestion",
    "FactualQuizQuestion",
    "ReasoningQuizQuestion",
    "LLMValidator",
    "LLMValidatorReport",
    "QuizResult",
    "quiz_report_summary",
]
