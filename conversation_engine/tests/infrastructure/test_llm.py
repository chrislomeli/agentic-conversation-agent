"""
Tests for LLM protocol and stub implementation.

Covers:
- LLMRequest / LLMResponse creation
- CallLLM protocol satisfaction
- call_llm_stub: no findings, with findings
- Protocol runtime checking
"""

import pytest

from conversation_engine.infrastructure.llm import (
    CallLLM,
    LLMRequest,
    LLMResponse,
    call_llm_stub,
)

# ── LLMRequest / LLMResponse ───────────────────────────────────────


class TestLLMRequest:
    def test_creation(self):
        req = LLMRequest(
            system_prompt="You are a test assistant.",
            user_message="Hello",
        )
        assert req.system_prompt == "You are a test assistant."
        assert req.user_message == "Hello"
        assert req.temperature == 0.2
        assert req.context == {}

    def test_frozen(self):
        req = LLMRequest(system_prompt="x", user_message="y")
        with pytest.raises(AttributeError):
            req.system_prompt = "changed"

    def test_with_context(self):
        req = LLMRequest(
            system_prompt="x",
            user_message="y",
            context={"findings": [{"id": "f1"}]},
        )
        assert req.context["findings"][0]["id"] == "f1"


class TestLLMResponse:
    def test_success(self):
        resp = LLMResponse(
            content="All good.",
            model="gpt-4o-mini",
            success=True,
        )
        assert resp.success
        assert resp.content == "All good."
        assert resp.error is None

    def test_error(self):
        resp = LLMResponse(
            content="",
            success=False,
            error="API rate limit exceeded",
        )
        assert not resp.success
        assert resp.error == "API rate limit exceeded"

    def test_frozen(self):
        resp = LLMResponse(content="x")
        with pytest.raises(AttributeError):
            resp.content = "changed"


# ── call_llm_stub ──────────────────────────────────────────────────


class TestCallLLMStub:
    def test_no_findings(self):
        req = LLMRequest(
            system_prompt="You are a validator.",
            user_message="Check the architecture.",
            context={},
        )
        resp = call_llm_stub(req)

        assert resp.success
        assert resp.model == "stub"
        assert "passed" in resp.content.lower() or "complete" in resp.content.lower()

    def test_with_findings(self):
        req = LLMRequest(
            system_prompt="You are a validator.",
            user_message="Check the architecture.",
            context={
                "findings": [
                    {"message": "Goal has no requirements"},
                    {"message": "Missing component link"},
                ]
            },
        )
        resp = call_llm_stub(req)

        assert resp.success
        assert "2 issue" in resp.content
        assert "Goal has no requirements" in resp.content

    def test_satisfies_protocol(self):
        """call_llm_stub satisfies the CallLLM Protocol."""
        assert isinstance(call_llm_stub, CallLLM)

    def test_custom_callable_satisfies_protocol(self):
        """Any callable with the right signature satisfies CallLLM."""

        def my_llm(request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="custom", model="test")

        assert isinstance(my_llm, CallLLM)
