"""
OpenAI adapter — wraps ChatOpenAI into the CallLLM protocol.

Usage:
    from conversation_engine.infrastructure.llm.openai_adapter import make_openai_llm

    llm = make_openai_llm()                          # uses OPENAI_API_KEY env var
    llm = make_openai_llm(model="gpt-4o-mini")       # cheaper model
    llm = make_openai_llm(api_key="sk-...")           # explicit key

The returned callable satisfies CallLLM: (LLMRequest) -> LLMResponse.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from conversation_engine.infrastructure.llm.protocols import LLMRequest, LLMResponse


def make_openai_llm(
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> OpenAICallLLM:
    """
    Factory that returns a CallLLM-compatible callable backed by OpenAI.

    Parameters
    ----------
    model : str
        OpenAI model name (default: gpt-4o-mini for cost efficiency).
    api_key : str, optional
        Explicit API key.  Falls back to OPENAI_API_KEY env var.
    temperature : float, optional
        Default temperature.  Can be overridden per-request via LLMRequest.temperature.
    max_tokens : int, optional
        Default max tokens.  Can be overridden per-request.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY env var "
            "or pass api_key= to make_openai_llm()."
        )

    chat = ChatOpenAI(
        model=model,
        api_key=key,
        temperature=temperature if temperature is not None else 0.2,
        max_tokens=max_tokens,
    )
    return OpenAICallLLM(chat=chat, model_name=model)


class OpenAICallLLM:
    """
    CallLLM implementation backed by langchain_openai.ChatOpenAI.

    Converts LLMRequest → ChatOpenAI messages → LLMResponse.
    """

    def __init__(self, chat: ChatOpenAI, model_name: str = ""):
        self._chat = chat
        self._model_name = model_name

    def __call__(self, request: LLMRequest) -> LLMResponse:
        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.user_message))

        try:
            result = self._chat.invoke(messages)

            usage = {}
            if hasattr(result, "response_metadata"):
                token_usage = result.response_metadata.get("token_usage", {})
                if token_usage:
                    usage = {
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0),
                    }

            return LLMResponse(
                content=result.content,
                model=self._model_name,
                usage=usage,
                success=True,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self._model_name,
                success=False,
                error=str(e),
            )
