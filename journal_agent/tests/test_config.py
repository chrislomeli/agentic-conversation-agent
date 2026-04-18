"""Layer 2 tests — Config: settings.py and config_builder.py."""

import json

import pytest

from journal_agent.configure.config_builder import (
    AI_NAME,
    DEFAULT_EXPLANATION_DEPTH,
    DEFAULT_INTERESTS,
    DEFAULT_LEARNING_STYLE,
    DEFAULT_PET_PEEVES,
    DEFAULT_RECENT_MESSAGES_COUNT,
    DEFAULT_RESPONSE_STYLE,
    DEFAULT_RETRIEVED_HISTORY_COUNT,
    DEFAULT_RETRIEVED_HISTORY_DISTANCE,
    DEFAULT_SESSION_MESSAGES_COUNT,
    DEFAULT_TONE,
    HUMAN_NAME,
    _mask_secret,
    _redacted_settings_json,
    configure_environment,
    literals,
)
from journal_agent.configure.settings import (
    LLM_ROLE_CONFIG,
    LLMLabel,
    LLMModel,
    LLMProvider,
    Settings,
    get_settings,
    models,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def clear_settings_cache():
    """Guarantee a fresh Settings instance; restore cache state after the test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════════════════════
# get_settings — caching behaviour
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSettings:
    def test_returns_settings_instance(self, clear_settings_cache):
        assert isinstance(get_settings(), Settings)

    def test_defaults_without_env_file(self, clear_settings_cache):
        s = get_settings()
        assert s.llm_source == LLMProvider.STUB
        assert s.llm_model is None

    def test_is_cached_same_object_on_second_call(self, clear_settings_cache):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_fresh_instance(self, clear_settings_cache):
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2


# ═══════════════════════════════════════════════════════════════════════════════
# Settings.selected_model — API key injection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectedModel:
    def test_returns_none_when_llm_model_is_none(self):
        s = Settings()
        assert s.selected_model is None

    def test_injects_openai_api_key(self):
        s = Settings(
            openai_api_key="sk-test-key-12345678",
            llm_model=LLMModel(
                model="gpt-4o",
                key_label="openai_api_key",
                provider=LLMProvider.OPENAI,
            ),
        )
        resolved = s.selected_model
        assert resolved is not None
        assert resolved.api_key is not None
        assert resolved.api_key.get_secret_value() == "sk-test-key-12345678"

    def test_does_not_mutate_original_llm_model(self):
        model = LLMModel(
            model="gpt-4o", key_label="openai_api_key", provider=LLMProvider.OPENAI
        )
        s = Settings(openai_api_key="sk-test-key-99999999", llm_model=model)
        s.selected_model
        assert model.api_key is None  # original dataclass is unchanged

    def test_returns_none_key_when_matching_secret_is_absent(self):
        # Pass openai_api_key=None explicitly — __init__ kwargs take highest priority
        # in pydantic-settings, overriding env vars and dotenv files.
        s = Settings(
            openai_api_key=None,
            llm_model=LLMModel(
                model="gpt-4o",
                key_label="openai_api_key",
                provider=LLMProvider.OPENAI,
            ),
        )
        resolved = s.selected_model
        assert resolved.api_key is None


# ═══════════════════════════════════════════════════════════════════════════════
# models dict — completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelsDict:
    def test_has_entry_for_every_llm_label(self):
        for label in LLMLabel:
            assert label in models, f"Missing entry in models dict for {label}"

    def test_stub_entry_is_none(self):
        assert models[LLMLabel.STUB] is None

    def test_non_stub_entries_are_llm_model_instances(self):
        for label, model in models.items():
            if label != LLMLabel.STUB:
                assert isinstance(model, LLMModel), f"{label} should be an LLMModel"

    def test_all_non_stub_models_have_provider_set(self):
        for label, model in models.items():
            if label != LLMLabel.STUB:
                assert isinstance(model.provider, LLMProvider)

    def test_all_non_stub_models_have_non_empty_model_string(self):
        for label, model in models.items():
            if label != LLMLabel.STUB:
                assert model.model, f"{label} has empty model string"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM_ROLE_CONFIG — type contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMRoleConfig:
    def test_keys_are_strings(self):
        for role in LLM_ROLE_CONFIG:
            assert isinstance(role, str)

    def test_values_are_valid_llm_labels(self):
        for label in LLM_ROLE_CONFIG.values():
            assert isinstance(label, LLMLabel)

    def test_conversation_role_is_present(self):
        assert "conversation" in LLM_ROLE_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# configure_environment — integration smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigureEnvironment:
    def test_returns_settings(self, clear_settings_cache, monkeypatch, capsys):
        monkeypatch.setenv("AI_ENV_FILE", "")
        result = configure_environment()
        assert isinstance(result, Settings)

    def test_sets_llm_model_from_use_model_literal(self, clear_settings_cache, monkeypatch, capsys):
        monkeypatch.setenv("AI_ENV_FILE", "")
        settings = configure_environment()
        expected_label = literals["USE_MODEL"]
        expected_model = models[expected_label]
        assert settings.llm_model == expected_model

    def test_sets_llm_source_to_model_provider(self, clear_settings_cache, monkeypatch, capsys):
        monkeypatch.setenv("AI_ENV_FILE", "")
        settings = configure_environment()
        expected_provider = models[literals["USE_MODEL"]].provider
        assert settings.llm_source == expected_provider


# ═══════════════════════════════════════════════════════════════════════════════
# _mask_secret — pure function
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaskSecret:
    def test_empty_string_returns_empty(self):
        assert _mask_secret("") == ""

    def test_none_like_falsy_returns_empty(self):
        assert _mask_secret("") == ""

    def test_short_value_returns_stars(self):
        assert _mask_secret("abc") == "***"

    def test_exactly_eight_chars_returns_stars(self):
        assert _mask_secret("12345678") == "***"

    def test_nine_chars_shows_head_and_tail(self):
        result = _mask_secret("123456789")
        assert result.startswith("1234")
        assert result.endswith("6789")
        assert "..." in result

    def test_long_key_shows_first_and_last_four(self):
        result = _mask_secret("sk-abcdef12345678abcd")
        assert result.startswith("sk-a")
        assert result.endswith("abcd")


# ═══════════════════════════════════════════════════════════════════════════════
# _redacted_settings_json — output contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedactedSettingsJson:
    def test_returns_valid_json(self):
        s = Settings()
        output = _redacted_settings_json(s)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_openai_key_is_masked_when_present(self):
        s = Settings(openai_api_key="sk-abcdef12345678abcd")
        output = _redacted_settings_json(s)
        # The raw secret must never appear in the output
        assert "sk-abcdef12345678abcd" not in output
        # pydantic model_dump(mode='json') already redacts SecretStr to "**********"
        # before _mask_secret sees it, so the final value is a masked form of that
        parsed = json.loads(output)
        assert parsed["openai_api_key"] is not None

    def test_none_api_keys_produce_empty_mask(self):
        s = Settings(openai_api_key=None)
        output = _redacted_settings_json(s)
        parsed = json.loads(output)
        openai = parsed.get("openai_api_key")
        assert openai is None or openai == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Application defaults — completeness and non-null contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplicationDefaults:
    def test_numeric_defaults_are_positive_integers(self):
        assert isinstance(DEFAULT_RECENT_MESSAGES_COUNT, int) and DEFAULT_RECENT_MESSAGES_COUNT > 0
        assert isinstance(DEFAULT_SESSION_MESSAGES_COUNT, int) and DEFAULT_SESSION_MESSAGES_COUNT > 0
        assert isinstance(DEFAULT_RETRIEVED_HISTORY_COUNT, int) and DEFAULT_RETRIEVED_HISTORY_COUNT > 0
        assert isinstance(DEFAULT_RETRIEVED_HISTORY_DISTANCE, int) and DEFAULT_RETRIEVED_HISTORY_DISTANCE > 0

    def test_string_defaults_are_non_empty(self):
        for name, value in [
            ("DEFAULT_RESPONSE_STYLE", DEFAULT_RESPONSE_STYLE),
            ("DEFAULT_EXPLANATION_DEPTH", DEFAULT_EXPLANATION_DEPTH),
            ("DEFAULT_TONE", DEFAULT_TONE),
            ("DEFAULT_LEARNING_STYLE", DEFAULT_LEARNING_STYLE),
            ("HUMAN_NAME", HUMAN_NAME),
        ]:
            assert isinstance(value, str) and len(value) > 0, f"{name} is empty"

    def test_ai_name_is_none(self):
        assert AI_NAME is None

    def test_default_interests_is_non_empty_list_of_strings(self):
        assert isinstance(DEFAULT_INTERESTS, list)
        assert len(DEFAULT_INTERESTS) > 0
        assert all(isinstance(i, str) for i in DEFAULT_INTERESTS)

    def test_default_pet_peeves_is_non_empty_list_of_strings(self):
        assert isinstance(DEFAULT_PET_PEEVES, list)
        assert len(DEFAULT_PET_PEEVES) > 0
        assert all(isinstance(p, str) for p in DEFAULT_PET_PEEVES)
