#!/usr/bin/env python3
"""
config_builder.py — STEP 01: Environment setup & configuration

Load API keys from .env, configure logging, and enable LangSmith tracing.
This is the first thing every other step calls — get the environment ready
before touching the world engine or any agents.

The main() here is just a smoke-test: call configure_environment() and verify
the settings load without errors.
"""

import logging
import os
from typing import Dict

from settings import get_settings, Settings, LLMLabel, LLMModel, LLMProvider

models: Dict[LLMLabel, LLMModel|None] = {
    # OpenAI models
    LLMLabel.GPT_MINI: LLMModel(key_label="openai_api_key", provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
    LLMLabel.GPT_NANO: LLMModel(key_label="openai_api_key", provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
    LLMLabel.GPT: LLMModel(key_label="openai_api_key", provider=LLMProvider.OPENAI, model="gpt-4o"),
    # Anthropic models
    LLMLabel.CLAUDE_SONNET: LLMModel(key_label="anthropic_api_key", provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
    LLMLabel.CLAUDE_OPUS: LLMModel(key_label="anthropic_api_key", provider=LLMProvider.ANTHROPIC, model="claude-opus-4-6"),
    LLMLabel.HAIKU: LLMModel(key_label="anthropic_api_key", provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5"),
    # Ollama models (local development)
    LLMLabel.OLLAMA_LLAMA3: LLMModel(key_label="ollama_base_url", provider=LLMProvider.OLLAMA, model="llama3.2:latest"),
    # Stub (no LLM)
    LLMLabel.STUB: None
}


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _redacted_settings_json(settings: Settings) -> str:
    payload = settings.model_dump(mode="json")
    for key in ("anthropic_api_key", "openai_api_key", "langchain_api_key"):
        if key in payload:
            payload[key] = _mask_secret(payload.get(key, ""))
    llm_model = payload.get("llm_model")
    if isinstance(llm_model, dict) and "api_key" in llm_model:
        llm_model["api_key"] = _mask_secret(llm_model.get("api_key") or "")
    import json
    return json.dumps(payload, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
literals = {
    "AI_ENV_FILE": "/Users/chrislomeli/Source/SECRETS/.env",
    "USE_MODEL": LLMLabel.OLLAMA_LLAMA3
}

def configure_environment() -> Settings:
    """
    Load API keys, configure logging, set up LangSmith tracing.

    Think of this as: "open the door, turn on the lights, get your tools ready"
    """
    os.environ.setdefault("AI_ENV_FILE", literals["AI_ENV_FILE"])
    settings = get_settings()  # Load .env file (API keys, project names, etc.)
    connection = models.get(literals["USE_MODEL"], models[LLMLabel.GPT_MINI])
    settings.llm_model = connection
    settings.llm_source = connection.provider if connection else LLMProvider.STUB


    # Set up Python logging so we can see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-35s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Print what we're using (redacted)
    print(_redacted_settings_json(settings))
    return settings


