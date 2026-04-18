"""Layer 3 tests — ContextBuilder (configure/context_builder.py).

API: get_context(instruction: ContextSpecification,
                 session_messages=None,
                 recent_messages=None,
                 retrieved_fragments=None) -> list[BaseMessage]

Output order: [SystemMessage, *recent_messages, *session_messages]
"""

from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from journal_agent.configure.context_builder import (
    ContextBuildError,
    ContextBuilder,
    ContextTooLargeError,
    MissingStateError,
)
from journal_agent.model.session import ContextSpecification, Fragment, PromptKey, Tag


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fragment(content: str = "past thought", tags: tuple[str, ...] = ("general",)) -> Fragment:
    return Fragment(
        session_id="sess-1",
        content=content,
        exchange_ids=[],
        tags=[Tag(tag=t) for t in tags],
        timestamp=datetime.now(),
    )


def make_spec(**kwargs) -> ContextSpecification:
    return ContextSpecification(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Happy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestHappyPath:
    def test_first_message_is_system_message(self):
        builder = ContextBuilder()
        messages = builder.get_context(
            instruction=make_spec(),
            session_messages=[HumanMessage(content="hello")],
        )
        assert isinstance(messages[0], SystemMessage)

    def test_output_order_is_system_then_recent_then_session(self):
        builder = ContextBuilder()
        messages = builder.get_context(
            instruction=make_spec(),
            session_messages=[HumanMessage(content="session-msg")],
            recent_messages=[HumanMessage(content="recent-msg")],
            retrieved_fragments=[make_fragment("past thought")],
        )
        assert isinstance(messages[0], SystemMessage)
        assert messages[1].content == "recent-msg"
        assert messages[2].content == "session-msg"

    def test_system_message_wraps_prompt_in_instructions_xml(self):
        builder = ContextBuilder()
        messages = builder.get_context(instruction=make_spec())
        sys_content = messages[0].content
        assert "<instructions>" in sys_content
        assert "</instructions>" in sys_content

    def test_retrieved_context_is_embedded_in_system_message_xml(self):
        builder = ContextBuilder()
        messages = builder.get_context(
            instruction=make_spec(),
            retrieved_fragments=[make_fragment("remembered this")],
        )
        sys_content = messages[0].content
        assert "<retrieved_context>" in sys_content
        assert "</retrieved_context>" in sys_content
        assert "remembered this" in sys_content

    def test_empty_retrieved_fragments_omits_retrieved_context_block(self):
        builder = ContextBuilder()
        messages = builder.get_context(instruction=make_spec(), retrieved_fragments=[])
        assert "<retrieved_context>" not in messages[0].content

    def test_none_retrieved_fragments_omits_retrieved_context_block(self):
        builder = ContextBuilder()
        messages = builder.get_context(instruction=make_spec())
        assert "<retrieved_context>" not in messages[0].content

    def test_each_prompt_key_produces_non_empty_system_message(self):
        builder = ContextBuilder()
        for key in PromptKey:
            messages = builder.get_context(instruction=make_spec(prompt_key=key))
            assert isinstance(messages[0], SystemMessage)
            assert len(messages[0].content) > 0, f"Empty system message for {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# Token budget — pruning behaviour
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenBudget:
    def test_drops_retrieved_context_when_over_budget(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "short prompt",
        )
        builder = ContextBuilder()
        builder.max_tokens = 100
        messages = builder.get_context(
            instruction=make_spec(),
            session_messages=[HumanMessage(content="q")],
            retrieved_fragments=[make_fragment(content="x" * 1000)],
        )
        assert "<retrieved_context>" not in messages[0].content
        assert any(m.content == "q" for m in messages[1:])

    def test_drops_recent_messages_when_still_over_after_retrieved_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "short prompt",
        )
        builder = ContextBuilder()
        builder.max_tokens = 50
        messages = builder.get_context(
            instruction=make_spec(),
            session_messages=[HumanMessage(content="keep-me")],
            recent_messages=[HumanMessage(content="x" * 400)],
        )
        non_system = [m.content for m in messages[1:]]
        assert "keep-me" in non_system
        assert not any("x" * 100 in c for c in non_system)

    def test_raises_context_too_large_when_prompt_alone_blows_budget(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "x" * 10_000,
        )
        builder = ContextBuilder()
        builder.max_tokens = 100

        with pytest.raises(ContextTooLargeError) as exc_info:
            builder.get_context(instruction=make_spec())

        err = exc_info.value
        assert err.tokens > err.budget
        assert isinstance(err, ContextBuildError)

    def test_context_too_large_error_is_context_build_error(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "x" * 10_000,
        )
        builder = ContextBuilder()
        builder.max_tokens = 100
        with pytest.raises(ContextBuildError):
            builder.get_context(instruction=make_spec())


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction limits — ContextSpecification slicing is respected
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstructionLimits:
    def test_last_k_session_messages_limits_session_output(self):
        builder = ContextBuilder()
        spec = make_spec(last_k_session_messages=2)
        msgs = [HumanMessage(content=f"s{i}") for i in range(5)]
        result = builder.get_context(instruction=spec, session_messages=msgs)
        session_in_result = [m for m in result[1:] if m.content.startswith("s")]
        assert len(session_in_result) == 2
        assert session_in_result[0].content == "s3"
        assert session_in_result[1].content == "s4"

    def test_zero_last_k_session_messages_produces_no_session_messages(self):
        builder = ContextBuilder()
        spec = make_spec(last_k_session_messages=0)
        msgs = [HumanMessage(content="should-be-excluded")]
        result = builder.get_context(instruction=spec, session_messages=msgs)
        assert all(m.content != "should-be-excluded" for m in result)

    def test_top_k_retrieved_history_limits_fragments_in_system_message(self):
        builder = ContextBuilder()
        spec = make_spec(top_k_retrieved_history=1)
        fragments = [make_fragment(f"fragment-{i}") for i in range(3)]
        result = builder.get_context(instruction=spec, retrieved_fragments=fragments)
        sys_content = result[0].content
        assert "fragment-0" in sys_content
        assert "fragment-1" not in sys_content
        assert "fragment-2" not in sys_content


# ═══════════════════════════════════════════════════════════════════════════════
# Error paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrors:
    def test_missing_prompt_key_raises_missing_state_error(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: (_ for _ in ()).throw(KeyError("injected")),
        )
        builder = ContextBuilder()
        with pytest.raises(MissingStateError) as exc_info:
            builder.get_context(instruction=make_spec())
        assert "prompt:" in str(exc_info.value)
        assert isinstance(exc_info.value, ContextBuildError)

    def test_missing_state_error_is_context_build_error(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: (_ for _ in ()).throw(KeyError("injected")),
        )
        with pytest.raises(ContextBuildError):
            ContextBuilder().get_context(instruction=make_spec())


# ═══════════════════════════════════════════════════════════════════════════════
# Token counting
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenCounting:
    def test_estimate_string_tokens_is_chars_div_4(self):
        builder = ContextBuilder()
        assert builder.count_string_tokens("a" * 40) == 10

    def test_estimate_message_tokens_includes_per_message_overhead(self):
        builder = ContextBuilder()
        msg = HumanMessage(content="a" * 40)
        assert builder.count_message_tokens([msg]) == 15
        assert builder.count_message_tokens([msg, msg]) == 30

    def test_empty_message_list_is_zero_tokens(self):
        assert ContextBuilder().count_message_tokens([]) == 0

    def test_empty_string_is_zero_tokens(self):
        assert ContextBuilder().count_string_tokens("") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# State isolation — pruning must not mutate caller's lists
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateIsolation:
    def test_pruning_does_not_mutate_caller_session_messages(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "short",
        )
        builder = ContextBuilder()
        builder.max_tokens = 50
        session_msgs = [HumanMessage(content="x" * 200)]
        recent_msgs = [HumanMessage(content="y" * 200)]
        try:
            builder.get_context(
                instruction=make_spec(),
                session_messages=session_msgs,
                recent_messages=recent_msgs,
            )
        except ContextTooLargeError:
            pass
        assert len(session_msgs) == 1
        assert len(recent_msgs) == 1
