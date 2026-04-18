"""Layer 5 tests — LLM client and registry (comms/llm_client.py, comms/llm_registry.py)."""

from unittest.mock import MagicMock, call

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from journal_agent.comms.llm_client import LLMClient, create_llm_client
from journal_agent.comms.llm_registry import LLMRegistry, build_llm_registry, _resolve_model
from journal_agent.configure.settings import LLMLabel, LLMModel, LLMProvider, Settings


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_chat():
    """A MagicMock standing in for any LangChain chat model."""
    m = MagicMock()
    m.invoke.return_value = AIMessage(content="mocked response")
    m.with_structured_output.return_value = m
    return m


@pytest.fixture
def stub_client(mock_chat):
    return LLMClient(model="stub-model", client=mock_chat)


def _make_openai_model(model: str = "gpt-4o") -> LLMModel:
    return LLMModel(model=model, key_label="openai_api_key", provider=LLMProvider.OPENAI)


def _make_ollama_model(model: str = "llama3.2:latest") -> LLMModel:
    return LLMModel(model=model, key_label="ollama_base_url", provider=LLMProvider.OLLAMA)


# ═══════════════════════════════════════════════════════════════════════════════
# LLMClient — delegation contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMClient:
    def test_model_property_returns_model_name(self, stub_client):
        assert stub_client.model == "stub-model"

    def test_get_client_returns_underlying_client(self, stub_client, mock_chat):
        assert stub_client.get_client() is mock_chat

    def test_chat_delegates_to_invoke(self, stub_client, mock_chat):
        messages = [HumanMessage(content="hello")]
        result = stub_client.chat(messages)
        mock_chat.invoke.assert_called_once_with(messages)
        assert result.content == "mocked response"

    def test_chat_returns_ai_message(self, stub_client):
        result = stub_client.chat([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)

    def test_structured_calls_with_structured_output(self, stub_client, mock_chat):
        schema = MagicMock()
        runnable = stub_client.structured(schema)
        mock_chat.with_structured_output.assert_called_once_with(schema)
        assert runnable is mock_chat

    def test_chat_passes_messages_through_unchanged(self, stub_client, mock_chat):
        messages = [HumanMessage(content="a"), HumanMessage(content="b")]
        stub_client.chat(messages)
        assert mock_chat.invoke.call_args == call(messages)


# ═══════════════════════════════════════════════════════════════════════════════
# create_llm_client — factory function
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateLLMClient:
    def test_raises_value_error_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(
                provider=MagicMock(spec=LLMProvider, name="BOGUS"),
                api_key=None,
                model="x",
            )

    def test_creates_llm_client_for_openai(self, monkeypatch):
        mock_chat = MagicMock()
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: mock_chat)
        client = create_llm_client(LLMProvider.OPENAI, "sk-test", "gpt-4o")
        assert isinstance(client, LLMClient)
        assert client.model == "gpt-4o"

    def test_creates_llm_client_for_anthropic(self, monkeypatch):
        mock_chat = MagicMock()
        monkeypatch.setattr("langchain_anthropic.ChatAnthropic", lambda **kwargs: mock_chat)
        client = create_llm_client(LLMProvider.ANTHROPIC, "sk-ant-test", "claude-sonnet")
        assert isinstance(client, LLMClient)
        assert client.model == "claude-sonnet"

    def test_creates_llm_client_for_ollama(self, monkeypatch):
        mock_chat = MagicMock()
        monkeypatch.setattr("langchain_ollama.ChatOllama", lambda **kwargs: mock_chat)
        client = create_llm_client(
            LLMProvider.OLLAMA, None, "llama3.2:latest",
            base_url="http://localhost:11434",
        )
        assert isinstance(client, LLMClient)
        assert client.model == "llama3.2:latest"

    def test_ollama_uses_base_url(self, monkeypatch):
        captured = {}
        def fake_ollama(**kwargs):
            captured.update(kwargs)
            return MagicMock()
        monkeypatch.setattr("langchain_ollama.ChatOllama", fake_ollama)
        create_llm_client(LLMProvider.OLLAMA, None, "llama3", base_url="http://custom:9999")
        assert captured["base_url"] == "http://custom:9999"

    def test_ollama_falls_back_to_default_base_url_when_none(self, monkeypatch):
        captured = {}
        def fake_ollama(**kwargs):
            captured.update(kwargs)
            return MagicMock()
        monkeypatch.setattr("langchain_ollama.ChatOllama", fake_ollama)
        create_llm_client(LLMProvider.OLLAMA, None, "llama3", base_url=None)
        assert captured["base_url"] == "http://localhost:11434"


# ═══════════════════════════════════════════════════════════════════════════════
# LLMRegistry — get and fallback logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMRegistry:
    def _make_registry(self, roles: dict[str, LLMClient]) -> LLMRegistry:
        return LLMRegistry(_clients=roles)

    def test_get_returns_client_for_known_role(self, stub_client):
        registry = self._make_registry({"conversation": stub_client})
        assert registry.get("conversation") is stub_client

    def test_get_falls_back_to_conversation_for_unknown_role(self, mock_chat):
        conv = LLMClient(model="conv", client=mock_chat)
        registry = self._make_registry({"conversation": conv})
        result = registry.get("classifier")
        assert result is conv

    def test_get_raises_key_error_when_no_conversation_fallback(self):
        registry = self._make_registry({})
        with pytest.raises(KeyError, match="conversation"):
            registry.get("missing_role")

    def test_get_raises_key_error_for_missing_role_with_no_fallback(self, mock_chat):
        other = LLMClient(model="other", client=mock_chat)
        registry = self._make_registry({"extractor": other})
        with pytest.raises(KeyError):
            registry.get("classifier")

    def test_roles_property_returns_sorted_list(self, mock_chat):
        a = LLMClient(model="a", client=mock_chat)
        b = LLMClient(model="b", client=mock_chat)
        registry = self._make_registry({"zebra": a, "apple": b})
        assert registry.roles == ["apple", "zebra"]

    def test_roles_property_empty_when_no_clients(self):
        assert self._make_registry({}).roles == []


# ═══════════════════════════════════════════════════════════════════════════════
# _resolve_model — key injection logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveModel:
    def _settings(self, **kwargs) -> Settings:
        return Settings(openai_api_key=None, anthropic_api_key=None, **kwargs)

    def test_returns_none_when_role_not_in_config(self):
        result = _resolve_model("unknown", {}, {}, self._settings())
        assert result is None

    def test_returns_none_when_model_is_stub(self):
        result = _resolve_model(
            "conversation",
            {"conversation": LLMLabel.STUB},
            {LLMLabel.STUB: None},
            self._settings(),
        )
        assert result is None

    def test_injects_api_key_from_settings(self):
        settings = Settings(openai_api_key="sk-injected-key-12345")
        model = _make_openai_model()
        result = _resolve_model(
            "conversation",
            {"conversation": LLMLabel.GPT},
            {LLMLabel.GPT: model},
            settings,
        )
        assert result is not None
        assert result.api_key is not None

    def test_does_not_mutate_source_model(self):
        settings = Settings(openai_api_key="sk-injected-key-12345")
        model = _make_openai_model()
        _resolve_model("conversation", {"conversation": LLMLabel.GPT}, {LLMLabel.GPT: model}, settings)
        assert model.api_key is None  # original unchanged


# ═══════════════════════════════════════════════════════════════════════════════
# build_llm_registry — integration of resolver + factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildLLMRegistry:
    def _patch_providers(self, monkeypatch) -> MagicMock:
        mock_chat = MagicMock()
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: mock_chat)
        monkeypatch.setattr("langchain_anthropic.ChatAnthropic", lambda **kw: mock_chat)
        monkeypatch.setattr("langchain_ollama.ChatOllama", lambda **kw: mock_chat)
        return mock_chat

    def test_returns_llm_registry(self, monkeypatch):
        self._patch_providers(monkeypatch)
        settings = Settings(openai_api_key="sk-test-key-12345678")
        role_config = {"conversation": LLMLabel.GPT}
        models = {LLMLabel.GPT: _make_openai_model()}
        registry = build_llm_registry(settings, models, role_config)
        assert isinstance(registry, LLMRegistry)

    def test_registers_client_for_each_role(self, monkeypatch):
        self._patch_providers(monkeypatch)
        settings = Settings(openai_api_key="sk-test-key-12345678")
        role_config = {"conversation": LLMLabel.GPT, "classifier": LLMLabel.GPT}
        models = {LLMLabel.GPT: _make_openai_model()}
        registry = build_llm_registry(settings, models, role_config)
        assert "conversation" in registry.roles
        assert "classifier" in registry.roles

    def test_skips_stub_roles(self, monkeypatch):
        self._patch_providers(monkeypatch)
        settings = Settings()
        role_config = {"conversation": LLMLabel.STUB}
        models = {LLMLabel.STUB: None}
        registry = build_llm_registry(settings, models, role_config)
        assert "conversation" not in registry.roles

    def test_registry_get_works_after_build(self, monkeypatch):
        self._patch_providers(monkeypatch)
        settings = Settings(openai_api_key="sk-test-key-12345678")
        role_config = {"conversation": LLMLabel.GPT}
        models = {LLMLabel.GPT: _make_openai_model()}
        registry = build_llm_registry(settings, models, role_config)
        client = registry.get("conversation")
        assert isinstance(client, LLMClient)
