"""Tests for #9c — graph split + per-turn invocation pattern.

Two layers of coverage:

    1. ``parse_user_input`` — the runner's command parser. Each accepted
       form (plain text, /quit, /reflect, /recall, /save) plus whitespace
       and missing-arg edge cases.

    2. Graph builders — ``build_conversation_graph`` and
       ``build_end_of_session_graph`` compile with mock dependencies and
       expose the expected node set. The conversation graph must NOT
       contain ``get_user_input`` (the whole point of #9c).
"""

from unittest.mock import MagicMock

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage

from journal_agent.comms.commands import (
    ParsedInput,
    build_turn_input,
    parse_user_input,
)
from journal_agent.graph.journal_graph import (
    Node,
    build_conversation_graph,
    build_end_of_session_graph,
)
from journal_agent.model.session import UserCommandValue


# ── parse_user_input ─────────────────────────────────────────────────────────


def test_parse_plain_text_is_a_message_with_no_command():
    p = parse_user_input("hello there")
    assert p == ParsedInput(message="hello there")
    assert p.command == UserCommandValue.NONE
    assert p.quit is False


def test_parse_strips_surrounding_whitespace():
    p = parse_user_input("   hi   ")
    assert p.message == "hi"


def test_parse_quit_signals_session_exit():
    p = parse_user_input("/quit")
    assert p.quit is True
    assert p.command == UserCommandValue.NONE


def test_parse_reflect_sets_reflect_command_with_prompt():
    p = parse_user_input("/reflect")
    assert p.command == UserCommandValue.REFLECT
    assert p.message  # a generated prompt asking for patterns
    assert p.quit is False


def test_parse_recall_with_topic_carries_args_and_prompt():
    p = parse_user_input("/recall my goals")
    assert p.command == UserCommandValue.RECALL
    assert p.command_args == "my goals"
    assert "my goals" in p.message


def test_parse_recall_without_topic_uses_generic_prompt():
    p = parse_user_input("/recall")
    assert p.command == UserCommandValue.RECALL
    assert p.command_args == ""
    assert p.message != ""  # still produces a generic recall prompt


def test_parse_save_with_args_carries_args_and_no_message():
    p = parse_user_input("/save 3 my topic")
    assert p.command == UserCommandValue.SAVE
    assert p.command_args == "3 my topic"
    # /save does not introduce a conversational turn — no HumanMessage
    assert p.message == ""


def test_parse_save_without_args_carries_empty_args():
    p = parse_user_input("/save")
    assert p.command == UserCommandValue.SAVE
    assert p.command_args == ""
    assert p.message == ""


# ── build_turn_input ─────────────────────────────────────────────────────────


def test_turn_input_always_includes_session_id():
    """JournalState requires session_id, so every turn input must carry it.

    On the first invocation for a fresh thread_id the checkpointer has no
    prior state and the input dict IS the initial JournalState — missing
    session_id triggers a Pydantic ValidationError at __start__.
    """
    parsed = parse_user_input("hello")
    payload = build_turn_input(parsed, session_id="s1")
    assert payload["session_id"] == "s1"


def test_turn_input_for_plain_message_includes_session_messages():
    parsed = parse_user_input("hello there")
    payload = build_turn_input(parsed, session_id="s1")
    assert "session_messages" in payload
    assert isinstance(payload["session_messages"][0], HumanMessage)
    assert payload["session_messages"][0].content == "hello there"
    assert payload["user_command"] == UserCommandValue.NONE


def test_turn_input_for_save_omits_session_messages():
    """/save shouldn't add an empty HumanMessage to the transcript."""
    parsed = parse_user_input("/save 3 my topic")
    payload = build_turn_input(parsed, session_id="s1")
    assert "session_messages" not in payload
    assert payload["user_command"] == UserCommandValue.SAVE
    assert payload["user_command_args"] == "3 my topic"


def test_turn_input_clears_system_message():
    """Each turn explicitly clears any prior system_message so it doesn't
    re-print on subsequent turns when the runner reads aget_state."""
    payload = build_turn_input(parse_user_input("hi"), session_id="s1")
    assert payload["system_message"] is None


# ── graph builders ───────────────────────────────────────────────────────────


def _mock_registry():
    """LLMRegistry stand-in whose .get(role) returns a usable mock."""
    reg = MagicMock()
    reg.get.return_value = MagicMock()
    return reg


def test_conversation_graph_has_expected_nodes_and_no_user_input():
    graph = build_conversation_graph(
        registry=_mock_registry(),
        session_store=MagicMock(),
        fragment_store=MagicMock(),
        insights_store=MagicMock(),
        profile_store=MagicMock(),
        reflection_graph=MagicMock(),
        checkpointer=MemorySaver(),
    )

    nodes = set(graph.get_graph().nodes)
    expected = {
        Node.INTENT_CLASSIFIER,
        Node.PROFILE_SCANNER,
        Node.RETRIEVE_HISTORY,
        Node.GET_AI_RESPONSE,
        Node.REFLECT,
        Node.RECALL,
        Node.CAPTURE,
    }
    missing = expected - nodes
    assert not missing, f"Conversation graph missing nodes: {missing}"

    # The whole point of #9c — get_user_input is gone.
    assert "get_user_input" not in nodes


def test_eos_graph_is_single_node():
    """After #1: the EOS graph has exactly one pipeline node, not seven.

    The pipeline is linear ETL with no branching.  A 7-node graph was the
    wrong shape for it.  All phases run sequentially inside end_of_session.
    """
    graph = build_end_of_session_graph(
        registry=_mock_registry(),
        fragment_store=MagicMock(),
        transcript_store=MagicMock(),
        thread_store=MagicMock(),
        classified_thread_store=MagicMock(),
        checkpointer=MemorySaver(),
    )

    nodes = set(graph.get_graph().nodes)

    # The single pipeline node must be present.
    assert Node.END_OF_SESSION in nodes, f"EOS graph missing end_of_session node; got: {nodes}"

    # The old 7 nodes must NOT appear as graph nodes — they are now phases
    # called inside end_of_session, not graph nodes in their own right.
    former_nodes = {
        "save_transcript", "exchange_decomposer", "save_threads",
        "thread_classifier", "save_classified_threads",
        "thread_fragment_extractor", "save_fragments",
    }
    leaked = former_nodes & nodes
    assert not leaked, f"Former EOS nodes leaked into graph structure: {leaked}"

    # Conversation-graph nodes must not appear either.
    assert Node.GET_AI_RESPONSE not in nodes
    assert Node.INTENT_CLASSIFIER not in nodes


def test_both_graphs_can_share_a_checkpointer():
    """A single checkpointer wires both graphs to the same thread state."""
    cp = MemorySaver()

    conv = build_conversation_graph(
        registry=_mock_registry(),
        session_store=MagicMock(),
        fragment_store=MagicMock(),
        insights_store=MagicMock(),
        profile_store=MagicMock(),
        reflection_graph=MagicMock(),
        checkpointer=cp,
    )
    eos = build_end_of_session_graph(
        registry=_mock_registry(),
        fragment_store=MagicMock(),
        transcript_store=MagicMock(),
        thread_store=MagicMock(),
        classified_thread_store=MagicMock(),
        checkpointer=cp,
    )

    assert conv is not None
    assert eos is not None
