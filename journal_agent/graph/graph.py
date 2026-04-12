import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter

from journal_agent.comms.human_chat import get_human_input
from journal_agent.comms.llm_client import LLMClient
from journal_agent.graph.state import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_PROCESSING,
    JournalState,
)
from journal_agent.model.turn import Role
from journal_agent.storage.api import SessionStore
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# ── Graph builder ─────────────────────────────────────────────────────────────

def node_trace(node_name: str | None = None):
    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        name = node_name or func.__name__

        @wraps(func)
        def wrapper(state: JournalState) -> dict:
            start = perf_counter()
            session_id = state.get("session_id", "unknown")
            try:
                result = func(state)
                elapsed = perf_counter() - start
                status = result.get("status") if isinstance(result, dict) else None
                if status == STATUS_ERROR:
                    logger.warning(
                        "Node %s completed with error in %.3fs (session_id=%s, status=%s, error_message=%s)",
                        name,
                        elapsed,
                        session_id,
                        status,
                        result.get("error_message") if isinstance(result, dict) else None,
                    )
                else:
                    logger.info(
                        "Node %s completed in %.3fs (session_id=%s, status=%s)",
                        name,
                        elapsed,
                        session_id,
                        status,
                    )
                return result
            except Exception:
                elapsed = perf_counter() - start
                logger.exception(
                    "Node %s failed in %.3fs (session_id=%s)",
                    name,
                    elapsed,
                    session_id,
                )
                raise

        return wrapper

    return decorator


def _make_get_user_input(session_store: SessionStore) -> Callable[..., dict]:
    @node_trace("get_user_input")
    def get_user_input(state: JournalState) -> dict:

        try:
            # Prompt user for input
            user_input = get_human_input()
            if user_input == "/quit":
                return {"status": STATUS_COMPLETED}

            # add input to session store
            session_store.cache_turn(
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


def _make_get_ai_response(llm: LLMClient, session_store: SessionStore) -> Callable[..., dict]:
    @node_trace("get_ai_response")
    def get_ai_response(state: JournalState) -> dict:
        try:
            messages = state["seed_context"] + state["session_messages"]
            response = llm.chat(messages)  # model answers using context
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            session_store.cache_turn(
                session_id=state["session_id"],
                role=Role.AI,
                content=content,
            )
            print(content)
            return {
                "session_messages": [AIMessage(content=content)],
                "status": STATUS_PROCESSING,
            }
        except Exception as e:
            logger.exception("Failed to generate AI response")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return get_ai_response


def _make_save_turn(session_store: SessionStore) -> Callable[..., dict]:
    @node_trace("save_turn")
    def save_turn(state: JournalState) -> dict:
        try:
            session_store.store_cache(state["session_id"])
            return {}
        except Exception as e:
            logger.exception("Failed to save turn information to store")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_turn


def route_on_user_input(state: JournalState) -> str:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Routing to end after user input error (session_id=%s, error_message=%s)",
            state.get("session_id", "unknown"),
            state.get("error_message"),
        )
        return END
    elif state["status"] == STATUS_COMPLETED:
        return "save_turn"
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
    llm: LLMClient,
    session_store: SessionStore,
):
    """Build and compile the journal conversation graph."""
    # noinspection PyTypeChecker
    builder = StateGraph(JournalState)  # no_qa

    builder.add_node("get_user_input", _make_get_user_input(session_store=session_store))
    builder.add_node("get_ai_response", _make_get_ai_response(llm=llm, session_store=session_store))
    builder.add_node("save_turn", _make_save_turn(session_store=session_store))

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_on_user_input)
    builder.add_conditional_edges("get_ai_response", route_on_ai_response)
    builder.add_edge("save_turn", END)
    compiled = builder.compile()
    return compiled
