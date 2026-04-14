import logging
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from journal_agent.comms.human_chat import get_human_input
from journal_agent.comms.llm_client import LLMClient
from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifier import make_exchange_classifier, make_fragment_extractor
from journal_agent.graph.nodes.save_data import make_save_transcript, make_save_exchanges, make_save_fragments
from journal_agent.graph.state import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_PROCESSING,
    JournalState,
)
from journal_agent.model.session import Role
from journal_agent.storage.exchange_store import TranscriptStore

logger = logging.getLogger(__name__)

# ── Graph builder ─────────────────────────────────────────────────────────────



def make_get_user_input(session_store: TranscriptStore) -> Callable[..., dict]:
    @node_trace("get_user_input")
    def get_user_input(state: JournalState) -> dict:

        try:
            # Prompt user for input
            user_input = get_human_input()
            if user_input == "/quit":
                return {"status": STATUS_COMPLETED}

            # add input to session store
            session_store.on_human_turn(
                session_id=state["session_id"], role=Role.HUMAN, content=user_input
            )

            # update status to processing
            return {
                "session_messages": [HumanMessage(content=user_input)],
                "status": STATUS_PROCESSING,
            }
        except KeyboardInterrupt:
            return {"status": STATUS_COMPLETED}
        except Exception as e:
            logger.exception("Failed to read user input")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return get_user_input


def make_get_ai_response(llm: LLMClient, session_store: TranscriptStore) -> Callable[..., dict]:
    @node_trace("get_ai_response")
    def get_ai_response(state: JournalState) -> dict:
        try:
            messages = state["seed_context"] + state["session_messages"]
            response = llm.chat(messages)  # model answers using context
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            exchange = session_store.on_ai_turn(
                session_id=state["session_id"],
                role=Role.AI,
                content=content,
            )
            print(content)
            return {
                "session_messages": [AIMessage(content=content)],
                "transcript" : [exchange],
                "status": STATUS_PROCESSING,
            }
        except Exception as e:
            logger.exception("Failed to generate AI response")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return get_ai_response

#
# def make_save_turn(session_store: TranscriptStore) -> Callable[..., dict]:
#     @node_trace("save_turn")
#     def save_turn(state: JournalState) -> dict:
#         try:
#             session_store.store_cache(state["session_id"])
#             return {}
#         except Exception as e:
#             logger.exception("Failed to save turn information to store")
#             return {
#                 "status": STATUS_ERROR,
#                 "error_message": str(e),
#             }
#
#     return save_turn


def route_on_user_input(state: JournalState) -> str:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Routing to end after user input error (session_id=%s, error_message=%s)",
            state.get("session_id", "unknown"),
            state.get("error_message"),
        )
        return END
    elif state["status"] == STATUS_COMPLETED:
        return "save_transcript"
    return "get_ai_response"


def route_on_ai_response(state: JournalState) -> str:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Routing to end after AI response error (session_id=%s, error_message=%s)",
            state.get("session_id", "unknown"),
            state.get("error_message"),
        )
        return END

    return "get_user_input"



def build_journal_graph(
    registry: LLMRegistry,
    session_store: TranscriptStore,
):
    """Build and compile the journal conversation graph."""
    conversation_llm = registry.get("conversation")

    # noinspection PyTypeChecker
    builder = StateGraph(JournalState)  # no_qa

    builder.add_node("get_user_input", make_get_user_input(session_store=session_store))
    builder.add_node("get_ai_response", make_get_ai_response(llm=conversation_llm, session_store=session_store))
    builder.add_node("exchange_classifier", make_exchange_classifier(llm=conversation_llm, session_store=session_store))
    builder.add_node("fragment_extractor", make_fragment_extractor(llm=conversation_llm))
    builder.add_node("save_transcript", make_save_transcript())
    builder.add_node("save_exchanges", make_save_exchanges())
    builder.add_node("save_fragments", make_save_fragments())


    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_on_user_input)
    builder.add_conditional_edges("get_ai_response", route_on_ai_response)
    builder.add_edge("save_transcript", "exchange_classifier")
    builder.add_edge("exchange_classifier", "save_exchanges")
    builder.add_edge("save_exchanges", "fragment_extractor")
    builder.add_edge("fragment_extractor", "save_fragments")
    builder.add_edge("save_fragments", END)
    compiled = builder.compile()
    return compiled
