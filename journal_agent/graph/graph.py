import json
import logging
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from journal_agent.comms.human_chat import get_human_input
from journal_agent.comms.llm_client import LLMClient
from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifier import (
    make_exchange_decomposer,
    make_thread_classifier,
    make_thread_fragment_extractor,
)
from journal_agent.graph.nodes.save_data import (
    make_save_transcript,
    make_save_threads,
    make_save_classified_threads,
    make_save_fragments_to_json, make_save_fragments_to_vectordb,
)
from journal_agent.graph.state import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_PROCESSING,
    JournalState,
)
from journal_agent.model.session import Role
from journal_agent.storage.exchange_store import TranscriptStore
from journal_agent.storage.vector_store import get_vector_store, VectorStore

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


from langchain_core.messages import HumanMessage


def make_retrieve_history(vector_store: VectorStore) -> Callable[..., dict]:
    @node_trace("retrieve_history")
    def retrieve_history(state: JournalState) -> dict:
        # Find the most recent HumanMessage, not just the last message.
        # Robust to future graph wiring that may interleave AI/tool messages.
        query_msg = next(
            (m for m in reversed(state["session_messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if query_msg is None:
            return {
                "retrieved_history": []}  # nothing to query against

        history = vector_store.search_fragments(query_msg.content, max_distance=1.3, top_k=5)
        return {"retrieved_history": history}

    return retrieve_history


def make_get_ai_response(llm: LLMClient, session_store: TranscriptStore) -> Callable[..., dict]:
    @node_trace("get_ai_response")
    def get_ai_response(state: JournalState) -> dict:
        try:
            # Build the system message with retrieved context baked in
            system_prompt = get_prompt("conversation")
            fragments = state["retrieved_history"]
            system_content = (
                f"{system_prompt}\n\nPREVIOUS CONVERSATIONS:\n\t"
                + json.dumps([{"content": f.content, "tag": [t.tag for t in f.tags]} for f in fragments])
            )
            system = [SystemMessage(system_content)]

            # construct the  prompt messages
            messages = state["seed_context"] + state["session_messages"]

            # get the llm response
            client = llm.get_client()
            response = client.invoke(system + messages)

            # model answers using context
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )

            # store the whole exchange
            exchange = session_store.on_ai_turn(
                session_id=state["session_id"],
                role=Role.AI,
                content=content,
            )
            print(content)


            # update the transcript with this exchange
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
    return "retrieve_history"


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
    vector_store: VectorStore
):
    """Build and compile the journal conversation graph.

    End-of-session classification pipeline (runs after user quits):
      save_transcript
        → exchange_decomposer       (transcript → threads)
        → save_threads
        → thread_classifier         (threads → classified_threads with tags)
        → save_classified_threads
        → thread_fragment_extractor (classified_threads → fragments)
        → save_fragments
        → END
    """
    conversation_llm = registry.get("conversation")
    classifier_llm = registry.get("classifier")
    extractor_llm = registry.get("extractor")

    # noinspection PyTypeChecker
    builder = StateGraph(JournalState)  # no_qa

    # Conversation loop nodes
    builder.add_node("get_user_input", make_get_user_input(session_store=session_store))
    builder.add_node("get_ai_response", make_get_ai_response(llm=conversation_llm, session_store=session_store))
    builder.add_node("retrieve_history", make_retrieve_history(vector_store=vector_store))

    # End-of-session classification pipeline (decomposed: 3 LLM stages)
    builder.add_node("exchange_decomposer", make_exchange_decomposer(llm=classifier_llm))
    builder.add_node("thread_classifier", make_thread_classifier(llm=classifier_llm))
    builder.add_node("thread_fragment_extractor", make_thread_fragment_extractor(llm=extractor_llm))
    builder.add_node("save_fragments_to_vectordb", make_save_fragments_to_vectordb(vector_store=vector_store))

    # Persistence nodes (one per pipeline artifact)
    builder.add_node("save_transcript", make_save_transcript())
    builder.add_node("save_threads", make_save_threads())
    builder.add_node("save_classified_threads", make_save_classified_threads())
    builder.add_node("save_fragments_to_json", make_save_fragments_to_json())

    # Wiring
    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_on_user_input)
    builder.add_conditional_edges("get_ai_response", route_on_ai_response)

    # Linear end-of-session pipeline
    builder.add_edge("retrieve_history", "get_ai_response")
    builder.add_edge("save_transcript", "exchange_decomposer")
    builder.add_edge("exchange_decomposer", "save_threads")
    builder.add_edge("save_threads", "thread_classifier")
    builder.add_edge("thread_classifier", "save_classified_threads")
    builder.add_edge("save_classified_threads", "thread_fragment_extractor")
    builder.add_edge("thread_fragment_extractor", "save_fragments_to_json")
    builder.add_edge("save_fragments_to_json", "save_fragments_to_vectordb")
    builder.add_edge("save_fragments_to_vectordb", END)
    compiled = builder.compile()
    return compiled
