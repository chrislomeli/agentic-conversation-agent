"""graph.py — Build and wire the LangGraph for the journal agent.

Two execution paths share a single compiled graph:

    Conversation loop (repeats every turn):
        get_user_input → intent_classifier → [retrieve_history] → get_ai_response
                  ↑                                                         │
                  └─────────────────────────────────────────────────┘

    End-of-session pipeline (runs once after /quit):
        save_transcript → exchange_decomposer → save_threads
          → thread_classifier → save_classified_threads
          → thread_fragment_extractor → save_fragments → END

Routing functions inspect ``JournalState.status`` to decide the next node.
On ERROR, every route sends the graph to END.
On COMPLETED (user typed /quit), the conversation loop exits into the
end-of-session pipeline.
"""

import logging
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from journal_agent.comms.human_chat import get_human_input
from journal_agent.comms.llm_client import LLMClient
from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.configure.context_builder import ContextBuilder, ContextBuildError
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifier import (
    make_exchange_decomposer,
    make_thread_classifier,
    make_thread_fragment_extractor, make_intent_classifier, make_profile_scanner,
)
from journal_agent.graph.nodes.save_data import (
    make_save_transcript,
    make_save_threads,
    make_save_classified_threads,
    make_save_fragments,
)
from journal_agent.graph.state import (
    JournalState,
)
from journal_agent.model.session import Role, Status
from journal_agent.repository import (
    PgFragmentRepository,
    TranscriptRepository,
    ThreadsRepository,
    UserProfileRepository,
    TranscriptStore,
)

logger = logging.getLogger(__name__)


# ── Graph builder ─────────────────────────────────────────────────────────────


def make_get_user_input(session_store: TranscriptStore) -> Callable[..., dict]:
    """Factory: node that reads console input, records the human turn, and
    returns a HumanMessage to append to session_messages."""
    @node_trace("get_user_input")
    def get_user_input(state: JournalState) -> dict:

        try:
            # Prompt user for input
            user_input = get_human_input()
            if user_input == "/quit":
                return {"status": Status.COMPLETED}

            # add input to session store
            session_store.on_human_turn(
                session_id=state["session_id"], role=Role.HUMAN, content=user_input
            )

            # update status to processing
            return {
                "session_messages": [HumanMessage(content=user_input)],
                "status": Status.PROCESSING,
            }
        except KeyboardInterrupt:
            return {"status": Status.COMPLETED}
        except Exception as e:
            logger.exception("Failed to read user input")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return get_user_input


def make_retrieve_history(fragment_store: PgFragmentRepository, context_builder: ContextBuilder | None = None) -> Callable[..., dict]:
    """Factory: node that queries the fragment store for fragments similar
    to the latest human message, enriched with intent-derived tags."""
    context_builder = context_builder or ContextBuilder()
    @node_trace("retrieve_history")
    def retrieve_history(state: JournalState) -> dict:
        try:
            # Find the most recent HumanMessage, not just the last message.
            # Robust to future graph wiring that may interleave AI/tool messages.
            query_msg = next(
                (m for m in reversed(state["session_messages"]) if isinstance(m, HumanMessage)),
                None,
            )
            if query_msg is None:
                return {"retrieved_history": []}  # nothing to query against

            # sprinkle in any tags from the intent classifier
            query = query_msg.content + " tags: " + ",".join(state["context_specification"].tags)

            # get specifications for searching
            tspec = state["context_specification"]
            distance = tspec.distance_retrieved_history
            top_k = tspec.top_k_retrieved_history

            # perform the search
            matches = fragment_store.search_fragments(query, min_relevance=distance, top_k=top_k)
            return {"retrieved_history": [fragment for fragment, _ in matches]}
        except Exception as e:
            logger.exception("Failed to retrieve history")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return retrieve_history


def make_get_ai_response(llm: LLMClient, session_store: TranscriptStore, context_builder: ContextBuilder | None = None) -> Callable[..., dict]:
    """Factory: node that assembles context, calls the conversation LLM,
    records the AI turn, and prints the response to the console."""
    context_builder = context_builder or ContextBuilder()
    @node_trace("get_ai_response")
    def get_ai_response(state: JournalState) -> dict:
        try:

            # Build the system message with retrieved context baked in
            instruction = state["context_specification"]
            prompt = get_prompt(key=instruction.prompt_key, state=state)

            messages = context_builder.get_context(
                prompt=prompt,
                instruction=instruction,
                session_messages=state["session_messages"],
                recent_messages=state["recent_messages"],
                retrieved_fragments=state["retrieved_history"],
            )

            # get the llm response
            response = llm.chat(messages)

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
            logger.info("AI: %s", content)

            # update the transcript with this exchange
            return {
                "session_messages": [AIMessage(content=content)],
                "transcript": [exchange],
                "status": Status.PROCESSING,
            }
        except ContextBuildError as e:
            logger.warning("Context build failed: %s", e)
            return {"status": Status.ERROR, "error_message": str(e)}
        except Exception as e:
            logger.exception("Failed to generate AI response")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return get_ai_response

# ── Routing ──────────────────────────────────────────────────────────────────


def _route_base(state: JournalState, *, next_node: str, on_completion: str = END) -> str:
    """Shared routing logic: ERROR → END, COMPLETED → *on_completion*, else → *next_node*."""
    if state["status"] == Status.ERROR:
        logger.warning(
            "Routing to END (session_id=%s, error_message=%s)",
            state.get("session_id", "unknown"),
            state.get("error_message"),
        )
        return END
    if state["status"] == Status.COMPLETED:
        return on_completion
    return next_node


def goto(node: str, on_completion: str = END) -> Callable[..., str]:
    """Generic router factory: go to *node* on PROCESSING, *on_completion* on COMPLETED, END on ERROR."""
    def goto_node(state: JournalState) -> str:
        return _route_base(state, next_node=node, on_completion=on_completion)
    return goto_node


def route_on_user_input(state: JournalState) -> str:
    """After user input: ERROR → END, COMPLETED → save_transcript, else → intent_classifier."""
    return _route_base(state, next_node="intent_classifier", on_completion="save_transcript")


def route_on_intent(state: JournalState) -> str:
    """After intent classification: branch to profile_scanner, retrieve_history, or get_ai_response."""
    base = _route_base(state, next_node="get_ai_response", on_completion="save_transcript")
    if base != "get_ai_response":
        return base
    if not state["user_profile"].is_current:
        return "profile_scanner"
    if state["context_specification"].top_k_retrieved_history > 0:
        return "retrieve_history"
    return "get_ai_response"


def route_on_profile(state: JournalState) -> str:
    """After profile scanner: branch to retrieve_history or get_ai_response."""
    base = _route_base(state, next_node="get_ai_response", on_completion="save_transcript")
    if base != "get_ai_response":
        return base
    if state["context_specification"].top_k_retrieved_history > 0:
        return "retrieve_history"
    return "get_ai_response"


def build_journal_graph(
        registry: LLMRegistry,
        session_store: TranscriptStore,
        fragment_store: PgFragmentRepository,
        profile_store: UserProfileRepository,
        transcript_store: TranscriptRepository | None = None,
        thread_store: ThreadsRepository | None = None,
        classified_thread_store: ThreadsRepository | None = None,
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
    builder.add_node("retrieve_history", make_retrieve_history(fragment_store=fragment_store))

    # Classification pipeline (decomposed: 3 LLM stages)
    builder.add_node("exchange_decomposer", make_exchange_decomposer(llm=classifier_llm))
    builder.add_node("thread_classifier", make_thread_classifier(llm=classifier_llm))
    builder.add_node("thread_fragment_extractor", make_thread_fragment_extractor(llm=extractor_llm))
    builder.add_node("intent_classifier", make_intent_classifier(llm=classifier_llm))
    builder.add_node("profile_scanner", make_profile_scanner(llm=classifier_llm, profile_store=profile_store))

    # Persistence nodes (one per pipeline artifact)
    builder.add_node("save_transcript", make_save_transcript(store=transcript_store))
    builder.add_node("save_threads", make_save_threads(store=thread_store))
    builder.add_node("save_classified_threads", make_save_classified_threads(store=classified_thread_store))
    builder.add_node("save_fragments", make_save_fragments(fragment_store=fragment_store))

    # Wiring
    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_on_user_input)

    builder.add_conditional_edges("intent_classifier", route_on_intent)
    builder.add_conditional_edges("profile_scanner", route_on_profile)
    builder.add_conditional_edges("get_ai_response", goto("get_user_input"))

    builder.add_conditional_edges("retrieve_history", goto("get_ai_response", on_completion="save_transcript"))
    builder.add_conditional_edges("exchange_decomposer", goto("save_threads", on_completion="save_transcript"))
    builder.add_conditional_edges("thread_classifier", goto("save_classified_threads", on_completion="save_transcript"))

    # Linear end-of-session pipeline
    builder.add_conditional_edges("save_transcript", goto("exchange_decomposer"))
    builder.add_conditional_edges("save_classified_threads", goto("thread_fragment_extractor"))
    builder.add_conditional_edges("thread_fragment_extractor", goto("save_fragments"))
    builder.add_edge("save_fragments", END)
    compiled = builder.compile()
    return compiled
