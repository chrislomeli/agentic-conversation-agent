"""
llm_client.py — Unified LLM client abstraction

This module provides a single interface for calling different LLM providers
(OpenAI, Anthropic, Ollama). Your code talks to LLMClient and doesn't care
which provider is actually running underneath.

WHY: Each provider has different APIs. This abstraction hides those differences
so you can swap providers by changing one config value, not rewriting code.

HOW: Each provider gets its own concrete implementation that knows how to:
  1. Translate your unified format to their API
  2. Call their actual client
  3. Translate their response back to your unified format
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Any

from langchain_core.messages import AIMessage
from pydantic import SecretStr

from settings import LLMProvider


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


class LLMClient(ABC):
    """
    Abstract base class that all providers implement.

    Your code calls these methods. The concrete implementations (OpenAIClient,
    AnthropicClient, etc.) handle translating to/from the provider's API.
    """
    _model: str
    _client: Any

    def chat(self, messages) -> AIMessage:
        # call the API
        response: AIMessage = self._client.invoke(messages)

        # Add response to history
        return response


# ═══════════════════════════════════════════════════════════════════════════════
# CONCRETE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class OpenAIClient(LLMClient):

    def __init__(self, api_key: SecretStr, model: str = "gpt-4o"):
        self._model = model
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=0,
                         api_key=api_key)
        self._client = llm


class AnthropicClient(LLMClient):
    """
    Anthropic (Claude) implementation.

    WHAT: Wraps the Anthropic Python client
    WHY: Anthropic takes system as a separate parameter (not in messages)
    HOW: We extract the system param separately before calling Anthropic
    """

    def __init__(self, api_key: SecretStr, model: str = "claude-haiku-4-5"):
        self._model = model
        from langchain_anthropic import ChatAnthropic
        a = api_key.get_secret_value()
        llm = ChatAnthropic(model_name=model,
                            api_key=a,
                            temperature=0
                            )
        self._client = llm


class OllamaClient(LLMClient):
    """
    Ollama implementation.

    WHAT: Wraps Ollama's OpenAI-compatible API
    WHY: Ollama exposes an API that looks like OpenAI (mostly)
    HOW: Similar to OpenAI but with different base URL and response format
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self._model = model
        # Initialize the model (ensure Ollama is running locally)
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url
        )
        self._client = llm


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_llm_client(
        provider: LLMProvider,
        api_key: SecretStr,
        model: str,
        base_url: Optional[str] = None,
) -> LLMClient:
    """
    Create the right LLM client based on the provider.

    WHAT: Factory function that returns the correct concrete client
    WHY: Centralized place to instantiate clients based on provider
    HOW: Matches provider string to the right class and returns an instance

    Example:
        client = create_llm_client("anthropic", api_key="sk-...", model="claude-3-5-sonnet-20241022")
        response = await client.chat(messages=[...], system="You are helpful")
    """
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(api_key=api_key, model=model)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(api_key=api_key, model=model)
    elif provider == LLMProvider.OLLAMA:
        return OllamaClient(base_url=base_url or "http://localhost:11434", model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
