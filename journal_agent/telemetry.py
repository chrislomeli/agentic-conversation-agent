"""telemetry.py — LangChain callback handler for per-call cost and error telemetry.

Attach a ``TelemetryCallbackHandler`` to any LangGraph invocation via the
``config["callbacks"]`` key.  It fires on every LLM call within the graph —
conversation turn AND end-of-session pipeline — and emits one structured log
record per call with token counts and, on failure, the error detail.

Usage::

    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [TelemetryCallbackHandler()],
    }
    await graph.ainvoke(input, config=config)

Log fields (all records):
    node            — LangGraph node name, e.g. ``"intent_classifier"``
    model           — model name string from the provider response
    prompt_tokens   — tokens in the prompt / system + user messages
    completion_tokens — tokens in the model completion
    total_tokens    — sum (computed when not provided by the model)

Log fields (error records only):
    error           — str(exception)
"""

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


def _token_counts(llm_output: dict) -> dict[str, int]:
    """Normalise provider-specific token dicts to a common shape.

    OpenAI:    llm_output["token_usage"] → prompt_tokens, completion_tokens, total_tokens
    Anthropic: llm_output["usage"]       → input_tokens,  output_tokens
    """
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    completion = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    total = usage.get("total_tokens") or (prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


class TelemetryCallbackHandler(BaseCallbackHandler):
    """Logs token usage and LLM errors in structured ``extra={}`` fields.

    One instance per graph invocation is fine; the handler is stateless.
    """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        llm_output = response.llm_output or {}
        counts = _token_counts(llm_output)
        metadata = kwargs.get("metadata") or {}
        logger.info(
            "llm call completed",
            extra={
                "node": metadata.get("langgraph_node", "unknown"),
                "model": llm_output.get("model_name", "unknown"),
                **counts,
            },
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        logger.error(
            "llm call failed",
            extra={
                "node": metadata.get("langgraph_node", "unknown"),
                "error": str(error),
            },
        )
