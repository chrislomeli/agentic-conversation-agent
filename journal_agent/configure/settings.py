from __future__ import annotations

import dataclasses
import os
from enum import Enum
from functools import lru_cache
from typing import Optional

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
    OLLAMA_LLAMA3 = "ollama-llama2"
    STUB = "STUB"

@dataclasses.dataclass
class LLMModel:
    model: str
    key_label: str
    provider: LLMProvider
    api_key: Optional[SecretStr] = None


class Settings(BaseSettings):
    # ── LLM credentials ───────────────────────────────────────────────────────
    llm_source: LLMProvider = LLMProvider.STUB
    llm_model: Optional[LLMModel] = None
    anthropic_api_key: SecretStr = ""
    openai_api_key: SecretStr = ""
    ollama_base_url: str = "http://localhost:11434"

    @property
    def selected_model(self) -> Optional[LLMModel]:
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

@lru_cache
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    The cache means the .env file is read once per process.
    In tests, call get_settings.cache_clear() before patching env vars
    so a fresh Settings object is created.
    """
    return Settings()


