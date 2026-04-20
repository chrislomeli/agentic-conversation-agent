"""
llm_client.py — Unified LLM client abstraction

This module provides a single interface for calling different LLM providers
(OpenAI, Anthropic, Ollama). Your code talks to LLMClient and doesn't care
which provider is actually running underneath.

WHY: Each provider has different APIs. LangChain already normalizes the
invoke/response shape, so our wrapper just holds the underlying chat model
and forwards calls to it. The factory function is the single place that
knows how to build each provider.
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import SecretStr

from journal_agent.configure.settings import LLMProvider


# ═══════════════════════════════════════════════════════════════════════════════
# THE UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class LLMResponse:
    """
    Unified response format. Regardless of which provider you use,
    you always get back this same structure.
    """

    text: str
    model: str
    stop_reason: str


class LLMClient:
    """
    Thin wrapper over a LangChain chat model.

    WHAT: Holds a configured LangChain model and forwards `chat()` to it.
    WHY:  LangChain already gives us a uniform invoke(messages) -> AIMessage
          surface across providers. A single class is enough — no need for
          per-provider subclasses.
    HOW:  Callers use `create_llm_client(...)` to build one; nodes call
          `client.chat(messages)` and receive an AIMessage.
    """

    def __init__(self, model: str, client: Any):
        self._model = model
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def chat(self, messages) -> AIMessage:
        return self._client.invoke(messages)

    async def achat(self, messages) -> AIMessage:
        return await self._client.ainvoke(messages)

    def get_client(self):
        return self._client

    def structured(self, schema: type):
        """Return a runnable that outputs instances of *schema*."""
        return self._client.with_structured_output(schema)

    def astructured(self, schema: type):
        """Return an async-capable runnable that outputs instances of *schema*.

        The returned runnable exposes ``.ainvoke()`` for use with asyncio.
        """
        return self._client.with_structured_output(schema)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def create_llm_client(
    provider: LLMProvider,
    api_key: SecretStr | str | None,
    model: str,
    base_url: str | None = None,
) -> LLMClient:
    """
    Build an LLMClient for the given provider.

    WHAT: Returns a single LLMClient wrapping the provider-specific chat model.
    WHY:  Centralize the per-provider construction so the rest of the app
          works against one type.
    HOW:  Pick the right LangChain chat class, instantiate it, and wrap it.

    For Ollama, pass `base_url` (api_key is ignored).
    """
    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    elif provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        chat = ChatAnthropic(model_name=model, api_key=key, temperature=0)
    elif provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url or "http://localhost:11434",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return LLMClient(model=model, client=chat)
