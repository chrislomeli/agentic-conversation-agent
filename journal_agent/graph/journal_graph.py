"""journal_graph.py — Build and wire the LangGraph for the journal agent.

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

import asyncio
import logging
from collections.abc import Callable
from typing import Coroutine, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from journal_agent.comms.human_chat import get_human_input, talk_to_human, Speaker
from journal_agent.comms.llm_client import LLMClient
from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.configure.context_builder import ContextBuilder, ContextBuildError
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifiers import (
    make_exchange_decomposer,
    make_thread_classifier,
    make_thread_fragment_extractor, make_intent_classifier, make_profile_scanner,
)
from journal_agent.graph.nodes.stores import (
    make_save_transcript,
    make_save_threads,
    make_save_classified_threads,
    make_save_fragments,
)
from journal_agent.graph.routing import _route_base, goto
from journal_agent.graph.state import JournalState
from journal_agent.model.session import Role, StatusValue, ContextSpecification, UserCommandValue, Fragment, Tag
from journal_agent.stores import (
    PgFragmentRepository,
    TranscriptRepository,
    ThreadsRepository,
    UserProfileRepository,
    TranscriptStore,
)

logger = logging.getLogger(__name__)


class Node:
    GET_USER_INPUT         = "get_user_input"
    GET_AI_RESPONSE        = "get_ai_response"
    RETRIEVE_HISTORY       = "retrieve_history"
    INTENT_CLASSIFIER      = "intent_classifier"
    PROFILE_SCANNER        = "profile_scanner"
    EXCHANGE_DECOMPOSER    = "exchange_decomposer"
    THREAD_CLASSIFIER      = "thread_classifier"
    THREAD_FRAG_EXTRACTOR  = "thread_fragment_extractor"
    SAVE_TRANSCRIPT        = "save_transcript"
    SAVE_THREADS           = "save_threads"
    SAVE_CLASSIFIED        = "save_classified_threads"
    SAVE_FRAGMENTS         = "save_fragments"
    REFLECT                = "reflect"
    RECALL                 = "recall"
    CAPTURE                = "capture"


# ── Graph builder ─────────────────────────────────────────────────────────────


def make_get_user_input(session_store: TranscriptStore) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Factory: node that reads console input, records the human turn, and
    returns a HumanMessage to append to session_messages."""
    @node_trace("get_user_input")
    async def get_user_input(state: JournalState) -> dict:
        try:
            if msg := state.system_message:
                talk_to_human(msg)

            user_input = await asyncio.to_thread(get_human_input)

            if user_input == "/quit":
                return {"status": StatusValue.COMPLETED}
            if user_input == "/reflect":
                return {
                    "user_command": UserCommandValue.REFLECT,
                    "session_messages": [HumanMessage(content="Please share the patterns and insights you've observed from my recent journal entries.")],
                }
            if user_input.startswith("/recall"):
                """
                todo command args
                /reflect 
                /recall <topic>
                /capture <goback: int> <topic: str>  -- capture the last n exchanges and store as topic x
                /memo  <topic: str>  -- capture this exchange and store as topic x
                """
                parts = user_input.split(maxsplit=1)
                args = parts[1].strip() if len(parts) > 1 else ""
                return {
                    "user_command": UserCommandValue.RECALL,
                    "user_command_args": args,
                    "session_messages": [HumanMessage(content=f"Please recall what I've previously written about: {args}" if args else "Please recall my recent journal entries.")],
                }
            if user_input.startswith("/save"):
                parts = user_input.split(maxsplit=1)
                args = parts[1].strip() if len(parts) > 1 else ""
                return {
                    "user_command": UserCommandValue.SAVE,
                    "user_command_args": args,
                    "system_message": None,
                }

            # add input to session store
            session_store.on_human_turn(
                session_id=state.session_id, role=Role.HUMAN, content=user_input
            )

            # update status to processing
            return {
                "session_messages": [HumanMessage(content=user_input)],
                "status": StatusValue.PROCESSING,
                "system_message": None,
            }
        except KeyboardInterrupt:
            return {"status": StatusValue.COMPLETED}
        except Exception as e:
            logger.exception("Failed to read user input")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return get_user_input


def make_retrieve_history(fragment_store: PgFragmentRepository) -> Callable[..., dict]:
    """Factory: node that queries the fragment store for fragments similar
    to the latest human message, enriched with intent-derived tags."""
    @node_trace("retrieve_history")
    def retrieve_history(state: JournalState) -> dict:
        try:
            # Find the most recent HumanMessage, not just the last message.
            # Robust to future graph wiring that may interleave AI/tool messages.
            query_msg = next(
                (m for m in reversed(state.session_messages) if isinstance(m, HumanMessage)),
                None,
            )
            if query_msg is None:
                return {"retrieved_history": []}  # nothing to query against

            # sprinkle in any tags from the intent classifier
            query = query_msg.content + " tags: " + ",".join(state.context_specification.tags)

            # get specifications for searching
            tspec = state.context_specification
            distance = tspec.distance_retrieved_history
            top_k = tspec.top_k_retrieved_history

            # perform the search
            matches = fragment_store.search_fragments(query, min_relevance=distance, top_k=top_k)
            return {"retrieved_history": [fragment for fragment, _ in matches]}
        except Exception as e:
            logger.exception("Failed to retrieve history")
            return {
                "status": StatusValue.ERROR,
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
            instruction = state.context_specification
            prompt = get_prompt(key=instruction.prompt_key, state=state)

            messages = context_builder.get_context(
                prompt=prompt,
                instruction=instruction,
                session_messages=state.session_messages,
                recent_messages=state.recent_messages,
                retrieved_fragments=state.retrieved_history,
                insights=state.latest_insights,
            )

            # get the llm response
            response = llm.chat(messages)

            # model answers using context
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )

            # store the whole exchange
            exchange = session_store.on_ai_turn(
                session_id=state.session_id,
                role=Role.AI,
                content=content,
            )
            talk_to_human(content, Speaker.AI)
            logger.info("AI: %s", content)

            # update the transcript with this exchange
            return {
                "session_messages": [AIMessage(content=content)],
                "transcript": [exchange],
                "status": StatusValue.PROCESSING,
            }
        except ContextBuildError as e:
            logger.warning("Context build failed: %s", e)
            return {"status": StatusValue.ERROR, "error_message": str(e)}
        except Exception as e:
            logger.exception("Failed to generate AI response")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return get_ai_response

def make_reflect_node(reflection_graph) ->  Callable[..., Coroutine[Any, Any, dict]]:

    @node_trace("reflect_node")
    async def reflect_node(state: JournalState) -> dict:

        reflection_input = {
            "fetch_parameters": state.fetch_parameters,
            "fragments": [],
            "clusters": [],
            "insights": [],
            "verified_insights": [],
            "latest_insights": [],
            "status": StatusValue.IDLE,
            "error_message": None,
        }
        result = await reflection_graph.ainvoke(reflection_input)
        return {"latest_insights": result["latest_insights"], "status": StatusValue.PROCESSING}

    return reflect_node

def make_recall_node(fragment_store: PgFragmentRepository) -> Callable[..., dict]:
    """Factory: node that searches FragmentStore by topic and surfaces matches as retrieved history."""

    @node_trace("recall_node")
    def recall_node(state: JournalState) -> dict:
        try:
            topic = state.user_command_args or ""
            if not topic:
                return {"retrieved_history": [], "status": StatusValue.PROCESSING}
            matches = fragment_store.search_fragments(topic, top_k=10)
            fragments = [f for f, _ in matches]
            return {"retrieved_history": fragments, "status": StatusValue.PROCESSING}

        except Exception as e:
            logger.exception("Failed to recall fragments")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return recall_node

def _fragment_from_transcript(n: int, topic: str, session_id: str, exchanges: list) -> tuple[Fragment, str]:
    """Build a Fragment from the last n transcript exchanges. Returns (fragment, confirmation_message)."""
    from datetime import datetime
    selected = exchanges[-n:] if n <= len(exchanges) else exchanges
    content = "\n\n".join(
        f"Human: {e.human.content if e.human else ''}\nAI: {e.ai.content if e.ai else ''}"
        for e in selected
    )
    fragment = Fragment(
        session_id=session_id,
        content=f"{topic}\n\n{content}",
        exchange_ids=[],  # exchanges not yet persisted; content is embedded inline
        tags=[],
        timestamp=datetime.now(),
    )
    return fragment, f"Saved last {len(selected)} exchange(s) as '{topic}'."


def _fragment_from_inline(topic: str, content: str, session_id: str) -> tuple[Fragment, str]:
    """Build a Fragment from inline text typed by the user. Returns (fragment, confirmation_message)."""
    from datetime import datetime
    fragment = Fragment(
        session_id=session_id,
        content=content,
        exchange_ids=[],
        tags=[Tag(tag=topic, note="/save command" )],
        timestamp=datetime.now(),
    )
    return fragment, f"Saved note as '{topic}'."


def make_capture_node(fragment_store: PgFragmentRepository) -> Callable[..., dict]:
    """Factory: save exchanges or inline text as a named fragment in the vector store.

    Syntax:
        /save <n> <topic>       — last n exchanges from transcript
        /save <topic>           — last 1 exchange from transcript
        /save <topic> <text>    — inline text (no transcript lookup)
    """

    @node_trace("capture")
    def capture_node(state: JournalState) -> dict:
        try:
            args = (state.user_command_args or "").strip()
            parts = args.split(maxsplit=1)
            session_id = state.session_id

            if not parts:
                return {
                    "system_message": "Usage: /save [n] <topic>  or  /save <topic> <text>",
                    "user_command": UserCommandValue.NONE,
                    "user_command_args": None,
                }

            if len(parts) == 2 and parts[0].isdigit():
                # /save 3 topic — transcript mode
                exchanges = state.transcript
                if not exchanges:
                    return {
                        "system_message": "Nothing to save — no exchanges in this session yet.",
                        "user_command": UserCommandValue.NONE,
                        "user_command_args": None,
                    }
                fragment, msg = _fragment_from_transcript(int(parts[0]), parts[1].strip(), session_id, exchanges)

            elif len(parts) == 2:
                # /save topic some inline text — inline mode
                fragment, msg = _fragment_from_inline(parts[0].strip(), parts[1].strip(), session_id)

            else:
                # /save topic — last 1 exchange
                exchanges = state.transcript
                if not exchanges:
                    return {
                        "system_message": "Nothing to save — no exchanges in this session yet.",
                        "user_command": UserCommandValue.NONE,
                        "user_command_args": None,
                    }
                fragment, msg = _fragment_from_transcript(1, parts[0].strip(), session_id, exchanges)

            fragment_store.save_fragments([fragment])
            return {
                "system_message": msg,
                "user_command": UserCommandValue.NONE,
                "user_command_args": None,
                "status": StatusValue.PROCESSING,
            }
        except Exception as e:
            logger.exception("Failed to capture exchanges")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return capture_node


# ── Routing ──────────────────────────────────────────────────────────────────

def route_on_user_input(state: JournalState) -> str:
    """After user input: ERROR → END, COMPLETED → save_transcript, else → intent_classifier."""
    if state.user_command == UserCommandValue.REFLECT:
        return Node.REFLECT
    if state.user_command == UserCommandValue.RECALL:
        return Node.RECALL
    if state.user_command == UserCommandValue.SAVE:
        return Node.CAPTURE
    return _route_base(state, next_node=Node.INTENT_CLASSIFIER, on_completion=Node.SAVE_TRANSCRIPT)


def route_on_intent(state: JournalState) -> str:
    """After intent classification: branch to profile_scanner, retrieve_history, or get_ai_response."""
    base = _route_base(state, next_node=Node.GET_AI_RESPONSE, on_completion=Node.SAVE_TRANSCRIPT)
    if base != Node.GET_AI_RESPONSE:
        return base
    if not state.user_profile.is_current:
        return Node.PROFILE_SCANNER
    if state.context_specification.top_k_retrieved_history > 0:
        return Node.RETRIEVE_HISTORY
    return Node.GET_AI_RESPONSE


def route_on_profile(state: JournalState) -> str:
    """After profile scanner: branch to retrieve_history or get_ai_response."""
    base = _route_base(state, next_node=Node.GET_AI_RESPONSE, on_completion=Node.SAVE_TRANSCRIPT)
    if base != Node.GET_AI_RESPONSE:
        return base
    if state.context_specification.top_k_retrieved_history > 0:
        return Node.RETRIEVE_HISTORY
    return Node.GET_AI_RESPONSE


def build_journal_graph(
        registry: LLMRegistry,
        session_store: TranscriptStore,
        fragment_store: PgFragmentRepository,
        profile_store: UserProfileRepository,
        reflection_graph: CompiledStateGraph,
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
    builder.add_node(Node.GET_USER_INPUT,   make_get_user_input(session_store=session_store))
    builder.add_node(Node.GET_AI_RESPONSE,  make_get_ai_response(llm=conversation_llm, session_store=session_store))
    builder.add_node(Node.RETRIEVE_HISTORY, make_retrieve_history(fragment_store=fragment_store))

    # Classification pipeline
    builder.add_node(Node.EXCHANGE_DECOMPOSER,   make_exchange_decomposer(llm=classifier_llm))
    builder.add_node(Node.THREAD_CLASSIFIER,     make_thread_classifier(llm=classifier_llm))
    builder.add_node(Node.THREAD_FRAG_EXTRACTOR, make_thread_fragment_extractor(llm=extractor_llm))
    builder.add_node(Node.INTENT_CLASSIFIER,     make_intent_classifier(llm=classifier_llm))
    builder.add_node(Node.PROFILE_SCANNER,       make_profile_scanner(llm=classifier_llm, profile_store=profile_store))

    # Persistence nodes
    builder.add_node(Node.SAVE_TRANSCRIPT, make_save_transcript(store=transcript_store))
    builder.add_node(Node.SAVE_THREADS,    make_save_threads(store=thread_store))
    builder.add_node(Node.SAVE_CLASSIFIED, make_save_classified_threads(store=classified_thread_store))
    -

    # Command nodes
    builder.add_node(Node.REFLECT, make_reflect_node(reflection_graph))
    builder.add_node(Node.RECALL,  make_recall_node(fragment_store=fragment_store))
    builder.add_node(Node.CAPTURE, make_capture_node(fragment_store=fragment_store))

    # Wiring
    builder.add_edge(START, Node.GET_USER_INPUT)
    builder.add_conditional_edges(Node.GET_USER_INPUT, route_on_user_input,
        [Node.REFLECT, Node.RECALL, Node.CAPTURE, Node.INTENT_CLASSIFIER, Node.SAVE_TRANSCRIPT, END])

    builder.add_conditional_edges(Node.INTENT_CLASSIFIER, route_on_intent,
        [Node.GET_AI_RESPONSE, Node.PROFILE_SCANNER, Node.RETRIEVE_HISTORY, Node.SAVE_TRANSCRIPT, END])
    builder.add_conditional_edges(Node.PROFILE_SCANNER, route_on_profile,
        [Node.GET_AI_RESPONSE, Node.RETRIEVE_HISTORY, Node.SAVE_TRANSCRIPT, END])
    builder.add_conditional_edges(Node.GET_AI_RESPONSE, goto(Node.GET_USER_INPUT),
        [Node.GET_USER_INPUT, END])

    builder.add_conditional_edges(Node.RETRIEVE_HISTORY, goto(Node.GET_AI_RESPONSE, on_completion=Node.SAVE_TRANSCRIPT),
        [Node.GET_AI_RESPONSE, Node.SAVE_TRANSCRIPT, END])
    builder.add_conditional_edges(Node.REFLECT, goto(Node.GET_AI_RESPONSE, on_completion=Node.SAVE_TRANSCRIPT),
        [Node.GET_AI_RESPONSE, Node.SAVE_TRANSCRIPT, END])
    builder.add_conditional_edges(Node.RECALL, goto(Node.GET_AI_RESPONSE, on_completion=Node.SAVE_TRANSCRIPT),
        [Node.GET_AI_RESPONSE, Node.SAVE_TRANSCRIPT, END])
    builder.add_conditional_edges(Node.CAPTURE, goto(Node.GET_USER_INPUT, on_completion=Node.SAVE_TRANSCRIPT),
        [Node.GET_USER_INPUT, Node.SAVE_TRANSCRIPT, END])

    # Linear end-of-session pipeline
    builder.add_conditional_edges(Node.SAVE_TRANSCRIPT,        goto(Node.EXCHANGE_DECOMPOSER),        [Node.EXCHANGE_DECOMPOSER, END])
    builder.add_conditional_edges(Node.EXCHANGE_DECOMPOSER,    goto(Node.SAVE_THREADS),               [Node.SAVE_THREADS, END])
    builder.add_conditional_edges(Node.SAVE_THREADS,           goto(Node.THREAD_CLASSIFIER),          [Node.THREAD_CLASSIFIER, END])
    builder.add_conditional_edges(Node.THREAD_CLASSIFIER,      goto(Node.SAVE_CLASSIFIED),            [Node.SAVE_CLASSIFIED, END])
    builder.add_conditional_edges(Node.SAVE_CLASSIFIED,        goto(Node.THREAD_FRAG_EXTRACTOR),      [Node.THREAD_FRAG_EXTRACTOR, END])
    builder.add_conditional_edges(Node.THREAD_FRAG_EXTRACTOR,  goto(Node.SAVE_FRAGMENTS),             [Node.SAVE_FRAGMENTS, END])
    builder.add_edge(Node.SAVE_FRAGMENTS, END)

    compiled = builder.compile()
    return compiled
