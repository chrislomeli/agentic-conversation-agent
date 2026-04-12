

import logging
from typing import Literal, Callable

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, StateGraph

from journal_agent.storage import SessionStore
from journal_agent.comms.human_chat import get_human_input
from journal_agent.comms.llm_client import LLMClient
from journal_agent.model.storage import Role

logger = logging.getLogger(__name__)

# ── Graph builder ─────────────────────────────────────────────────────────────

from .state import JournalState, STATUS_ERROR, STATUS_COMPLETED, STATUS_PROCESSING


def _make_get_ai_response(
        llm: LLMClient,
        session_store: SessionStore ) -> Callable[..., dict]:

    def get_ai_response(state: JournalState) -> dict:
        messages = state["messages"]  # full history
        response = llm.chat(messages)  # model answers using context
        session_store.cache_turn(
            session_id=state["session_id"],
            role=Role.AI,
            content=response.content,
        )
        return {
            "messages": [AIMessage(content=response.content)],
            "status": STATUS_PROCESSING,
        }
    return get_ai_response


def _make_get_user_input(
        session_store: SessionStore )-> Callable[..., dict]:
    def get_user_input(state: JournalState) -> dict:

        # Prompt user for input
        user_input = get_human_input()
        if user_input == "/quit":
            return {"status": STATUS_COMPLETED}

        # add input to session store
        session_store.cache_turn(session_id=state["session_id"], role=Role.HUMAN, content=user_input)

        # update status to processing
        return {
            "messages": [HumanMessage(content=user_input)],
            "status": STATUS_PROCESSING}

    return get_user_input


def _make_save_turn(
        session_store: SessionStore )-> Callable[..., dict]:
    def save_turn(state: JournalState) -> dict:
        print(session_store.get_last_session_id())
        return {}

    return save_turn


def route_on_user_input(state: JournalState) -> Literal["get_ai_response", "save_turn", "__end__"]:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Warning goes here"
        )
        return "__end__"
    elif state["status"] == STATUS_COMPLETED:
        return "save_turn"
    return "get_ai_response"

def route_on_ai_response(state: JournalState) -> Literal["save_turn", "__end__"]:
    if state.status == "error":
        logger.warning(
            "Warning goes here"
        )
        return "__end__"

    return "save_turn"

def route_on_save_turn(state: JournalState) -> Literal["get_user_input", "__end__"]:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Warning goes here"
        )
        return "__end__"
    elif state["status"] == STATUS_COMPLETED:
        return "__end__"
    return "get_user_input"


def build_journal_graph(
        llm: LLMClient,
        session_store: SessionStore,
):
    """

    """
    # noinspection PyTypeChecker
    builder = StateGraph(JournalState) # no_qa

    builder.add_node("get_user_input", _make_get_user_input(session_store=session_store))
    builder.add_node("get_ai_response", _make_get_ai_response(llm=llm) )
    builder.add_node("save_turn", _make_save_turn(session_store=session_store))

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_on_user_input)
    builder.add_conditional_edges("get_ai_response", route_on_ai_response)
    builder.add_conditional_edges("save_turn", route_on_save_turn)
    compiled = builder.compile()
    return compiled
