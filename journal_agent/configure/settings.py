from __future__ import annotations

import dataclasses
import os
from enum import Enum
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(Enum):
    STUB = "STUB"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    OLLAMA = "OLLAMA"


class LLMLabel(Enum):
    GPT_MINI = "gpt-mini"
    GPT_NANO = "gpt-nano"
    GPT = "gpt"
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_OPUS = "claude-opus"
    HAIKU = "haiku"
    OLLAMA_LLAMA3 = "ollama-llama3"
    STUB = "STUB"


@dataclasses.dataclass
class LLMModel:
    model: str
    key_label: str
    provider: LLMProvider
    api_key: SecretStr | None = None


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────────────
    postgres_url: str = "postgresql://localhost:5432/journal"

    # ── LLM credentials ───────────────────────────────────────────────────────
    llm_source: LLMProvider = LLMProvider.STUB
    llm_model: LLMModel | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    @property
    def selected_model(self) -> LLMModel | None:
        if self.llm_model is None:
            return None
        connection = dataclasses.replace(self.llm_model)
        raw_secret = getattr(self, connection.key_label, None)
        connection.api_key = raw_secret or None
        return connection

    model_config = SettingsConfigDict(
        # AI_ENV_FILE=/path/to/.env for local dev.
        # Unset (None) on K8s — pydantic-settings skips file loading entirely
        # and reads from environment variables only.
        env_file=os.getenv("AI_ENV_FILE"),
        env_file_encoding="utf-8",
        # Silently ignore keys in the .env file that are not defined above.
        # Useful because the shared .env may contain keys for other projects.
        extra="ignore",
    )



# ── Available model definitions ────────────────────────────────────────────
models: dict[LLMLabel, LLMModel | None] = {
    # OpenAI models
    LLMLabel.GPT_MINI: LLMModel(
        key_label="openai_api_key", provider=LLMProvider.OPENAI, model="gpt-4o-mini"
    ),
    LLMLabel.GPT_NANO: LLMModel(
        key_label="openai_api_key", provider=LLMProvider.OPENAI, model="gpt-4o-mini"
    ),
    LLMLabel.GPT: LLMModel(key_label="openai_api_key", provider=LLMProvider.OPENAI, model="gpt-4o"),
    # Anthropic models
    LLMLabel.CLAUDE_SONNET: LLMModel(
        key_label="anthropic_api_key", provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"
    ),
    LLMLabel.CLAUDE_OPUS: LLMModel(
        key_label="anthropic_api_key", provider=LLMProvider.ANTHROPIC, model="claude-opus-4-6"
    ),
    LLMLabel.HAIKU: LLMModel(
        key_label="anthropic_api_key", provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5"
    ),
    # Ollama models (local development)
    LLMLabel.OLLAMA_LLAMA3: LLMModel(
        key_label="ollama_base_url", provider=LLMProvider.OLLAMA, model="llama3.2:latest"
    ),
    # Stub (no LLM)
    LLMLabel.STUB: None,
}

# ── Role → model label mapping ────────────────────────────────────────────
# Each entry says: "for this role, use this model from the models dict."
# If a role is absent here it falls back to the default ("conversation").
LLM_ROLE_CONFIG: dict[str, LLMLabel] = {
    "conversation": LLMLabel.OLLAMA_LLAMA3,
    "classifier": LLMLabel.GPT,
    "extractor": LLMLabel.GPT,
}


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    The cache means the .env file is read once per process.
    In tests, call get_settings.cache_clear() before patching env vars
    so a fresh Settings object is created.
    """
    return Settings()
