"""journal_graph.py — Build the journal agent's two compiled graphs.

After #9c the conversation no longer loops inside the graph. The runner
(terminal main.py or FastAPI endpoint) drives the loop in Python:

    parse user input → invoke conversation graph for ONE turn →
        consume token events → repeat. On /quit, invoke the EOS graph.

Conversation graph — one turn per invocation:

    START → (route_on_start)
       ├── REFLECT  ─┐
       ├── RECALL  ──┤
       ├── CAPTURE ──┼──→ END (CAPTURE has no AI turn)
       └── INTENT_CLASSIFIER ─→ [PROFILE_SCANNER] ─→ [RETRIEVE_HISTORY]
                                          ↘
                                        GET_AI_RESPONSE → END

End-of-session graph — linear ETL invoked once at /quit:

    START → SAVE_TRANSCRIPT → EXCHANGE_DECOMPOSER → SAVE_THREADS
        → THREAD_CLASSIFIER → SAVE_CLASSIFIED_THREADS
        → THREAD_FRAG_EXTRACTOR → SAVE_FRAGMENTS → END

Both graphs use the same ``JournalState`` schema and share a checkpointer
keyed by ``thread_id == session_id``. The EOS graph reads the final state
the conversation graph left in the checkpoint.

Routing functions inspect ``JournalState.status``:
    ERROR → route to END (every router).
    Otherwise route by user_command (start) or by classification flags
    (intent / profile).
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, Coroutine

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from journal_agent.comms.llm_client import LLMClient
from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.configure.context_builder import ContextBuilder, ContextBuildError
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifiers import (
    make_intent_classifier,
    make_profile_scanner,
)
from journal_agent.graph.nodes.eos_pipeline import make_end_of_session_node
from journal_agent.graph.routing import _route_base, goto
from journal_agent.graph.state import JournalState, ReflectionState
from journal_agent.model.session import Fragment, Role, StatusValue, Tag, UserCommandValue
from journal_agent.stores import (
    CaptureRepository,
    FragmentRepository,
    ThreadsRepository,
    TranscriptRepository,
    TranscriptStore,
    UserProfileRepository, InsightsRepository,
    SubjectsRepository,
)

logger = logging.getLogger(__name__)


class Node:
    # Conversation graph nodes
    GET_AI_RESPONSE = "get_ai_response"
    RETRIEVE_HISTORY = "retrieve_history"
    INTENT_CLASSIFIER = "intent_classifier"
    PROFILE_SCANNER = "profile_scanner"
    REFLECT = "reflect"
    REFLECT2 = "reflect2"
    RECALL = "recall"
    CAPTURE = "capture"
    CAPTURE_SEARCH = "capture_search"

    # End-of-session graph node (#1: collapsed 7-node chain into one)
    END_OF_SESSION = "end_of_session"


# ── Graph nodes ──────────────────────────────────────────────────────────────


def make_retrieve_history(fragment_store: FragmentRepository) -> Callable[..., dict]:
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


def make_get_ai_response(llm: LLMClient, session_store: TranscriptStore,
                         context_builder: ContextBuilder | None = None) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Factory: node that assembles context, streams the conversation LLM,
    and records the AI turn.

    The node calls ``llm.astream(messages)``; LangChain emits
    ``on_chat_model_stream`` events as chunks arrive, which the caller can
    consume via ``graph.astream_events(version="v2")``. The node itself
    accumulates chunks into the final AIMessage and persists the exchange
    on completion. Terminal printing happens *outside* the node so the same
    code serves both CLI and HTTP callers.
    """
    context_builder = context_builder or ContextBuilder()

    @node_trace("get_ai_response")
    async def get_ai_response(state: JournalState) -> dict:
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
                claim_insights=state.claim_insights or None,
            )

            # Stream the response. Each chunk emits an on_chat_model_stream
            # event that downstream consumers (terminal, SSE) can render.
            chunks: list = []
            async for chunk in llm.astream(messages):
                chunks.append(chunk)

            content = "".join(
                str(c.content) for c in chunks if isinstance(c.content, str)
            ).strip()

            # store the whole exchange
            exchange = session_store.on_ai_turn(
                session_id=state.session_id,
                role=Role.AI,
                content=content,
            )

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


def make_reflect_node(reflection_graph: CompiledStateGraph, fragment_store: FragmentRepository,
                      insights_store: InsightsRepository) -> Callable[..., Coroutine[Any, Any, dict]]:
    @node_trace("reflect_node")
    async def reflect_node(state: JournalState) -> dict:
        try:

            # ---Process any new insights ----------------
            cursor = datetime.min
            while True:
                fragments = fragment_store.load_unprocessed_fragments(after=cursor, limit=500)
                if not fragments:
                    break

                reflection_input = ReflectionState(
                    session_id=state.session_id,
                    fetch_parameters=state.fetch_parameters,
                    fragments=fragments,
                    clusters=[],
                    insights=[],
                    verified_insights=[],
                    latest_insights=[],
                    status=StatusValue.IDLE,
                    error_message=None
                )
                response = await reflection_graph.ainvoke(reflection_input)
                insights_store.save_insights(response["verified_insights"])
                cursor = fragments[-1].timestamp

            # -- now just get the latest insights --------------
            latest_insights = insights_store.load_insights() or []

            return {
                "latest_insights": latest_insights,
                "status": StatusValue.PROCESSING
            }

        except Exception as e:
            logger.exception("Failed to retrieve history")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return reflect_node


def make_claim_reflect_node(
    claim_reflection_graph: CompiledStateGraph,
    subjects_repo: SubjectsRepository,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Factory: node that runs the Phase 11 claim-based reflection pipeline.

    Fetches unprocessed fragments via SubjectsRepository, invokes the
    claim reflection graph, then summarises active subjects + traction
    as latest_insights for the AI response node to narrate.
    """
    from journal_agent.configure.prompts import get_prompt_version
    from journal_agent.model.session import PromptKey
    from journal_agent.graph.nodes.insight_nodes import compute_traction
    from journal_agent.model.insights import Stance, SubjectSnapshot

    @node_trace("claim_reflect_node")
    async def claim_reflect_node(state: JournalState) -> dict:
        try:
            model_sig = f"stance_classifier/{get_prompt_version(PromptKey.STANCE_CLASSIFIER)}"

            cursor = datetime.min
            while True:
                frag_ids = subjects_repo.fetch_unprocessed_fragment_ids(
                    model_signature=model_sig, after=cursor, limit=500,
                )
                if not frag_ids:
                    break

                fragments = []
                for fid in frag_ids:
                    rows = subjects_repo._pg.fetch_rows(
                        "SELECT * FROM fragments WHERE fragment_id = %s", (fid,),
                    )
                    if rows:
                        r = rows[0]
                        fragments.append(Fragment(
                            fragment_id=r["fragment_id"],
                            session_id=r.get("session_id", ""),
                            content=r["content"],
                            exchange_ids=[],
                            tags=[],
                            embedding=list(r["embedding"]) if r.get("embedding") is not None else [],
                            timestamp=r["timestamp"],
                        ))

                if not fragments:
                    break

                reflection_input = ReflectionState(
                    session_id=state.session_id,
                    fragments=fragments,
                    status=StatusValue.IDLE,
                )
                await claim_reflection_graph.ainvoke(reflection_input)
                cursor = fragments[-1].timestamp

            active_with_claims = subjects_repo.list_active_subjects_with_claims()
            claim_insights = []
            for subj, claim in active_with_claims:
                votes = subjects_repo.fetch_votes_for_subject(subj.subject_id)
                traction = compute_traction(votes)
                support = sum(1 for v in votes if v.stance == Stance.SUPPORT)
                contradict = sum(1 for v in votes if v.stance == Stance.CONTRADICT)
                claim_insights.append(SubjectSnapshot(
                    label=subj.label,
                    claim=claim.text,
                    traction=round(traction, 2),
                    support=support,
                    contradict=contradict,
                ))

            return {
                "system_message": f"Claim-based reflection complete. "
                                  f"{len(claim_insights)} active subject(s).",
                "claim_insights": claim_insights,
                "retrieved_history": [],
                "status": StatusValue.PROCESSING,
            }

        except Exception as e:
            logger.exception("Claim reflection failed")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return claim_reflect_node


def make_recall_node(fragment_store: FragmentRepository) -> Callable[..., dict]:
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
        tags=[Tag(tag=topic, note="/save command")],
        timestamp=datetime.now(),
    )
    return fragment, f"Saved note as '{topic}'."


def make_capture_search_node(capture_store: CaptureRepository) -> Callable[..., dict]:
    """Factory: search the captures table by topic and surface matches as retrieved history."""

    @node_trace("capture_search_node")
    def capture_search_node(state: JournalState) -> dict:
        try:
            topic = state.user_command_args or ""
            if not topic:
                return {"retrieved_history": [], "status": StatusValue.PROCESSING}
            matches = capture_store.search_captures(topic, top_k=10)
            fragments = [f for f, _ in matches]
            return {"retrieved_history": fragments, "status": StatusValue.PROCESSING}
        except Exception as e:
            logger.exception("Failed to search captures")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return capture_search_node


def make_capture_node(capture_store: CaptureRepository) -> Callable[..., dict]:
    """Factory: save exchanges or inline text as a named capture in the captures table.

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

            capture_store.save_captures([fragment])
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

def route_on_start(state: JournalState) -> str:
    """Initial dispatch from START based on the runner-supplied user_command.

    Command turns route directly to their command node; plain conversation
    turns route to the intent classifier.
    """
    if state.user_command == UserCommandValue.REFLECT:
        return Node.REFLECT
    if state.user_command == UserCommandValue.REFLECT2:
        return Node.REFLECT2
    if state.user_command == UserCommandValue.RECALL:
        return Node.RECALL
    if state.user_command == UserCommandValue.SAVE:
        return Node.CAPTURE
    if state.user_command == UserCommandValue.CAPTURE:
        return Node.CAPTURE_SEARCH
    return Node.INTENT_CLASSIFIER


def route_on_intent(state: JournalState) -> str:
    """After intent classification: profile_scanner, retrieve_history, or get_ai_response."""
    base = _route_base(state, next_node=Node.GET_AI_RESPONSE)  # on_completion → END
    if base != Node.GET_AI_RESPONSE:
        return base
    if not state.user_profile.is_current:
        return Node.PROFILE_SCANNER
    if state.context_specification.top_k_retrieved_history > 0:
        return Node.RETRIEVE_HISTORY
    return Node.GET_AI_RESPONSE


def route_on_profile(state: JournalState) -> str:
    """After profile scanner: retrieve_history or get_ai_response."""
    base = _route_base(state, next_node=Node.GET_AI_RESPONSE)
    if base != Node.GET_AI_RESPONSE:
        return base
    if state.context_specification.top_k_retrieved_history > 0:
        return Node.RETRIEVE_HISTORY
    return Node.GET_AI_RESPONSE


def build_conversation_graph(
        registry: LLMRegistry,
        session_store: TranscriptStore,
        fragment_store: FragmentRepository,
        insights_store: InsightsRepository,
        profile_store: UserProfileRepository,
        reflection_graph: CompiledStateGraph,
        capture_store: CaptureRepository | None = None,
        claim_reflection_graph: CompiledStateGraph | None = None,
        subjects_repo: SubjectsRepository | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the per-turn conversation graph.

    One invocation = one turn. The runner supplies user_command + the new
    HumanMessage in the input dict; the graph picks up prior state from the
    checkpointer, runs the appropriate path, and exits to END. The runner
    consumes token events via ``astream_events(version="v2")``.
    """
    conversation_llm = registry.get("conversation")
    classifier_llm = registry.get("classifier")

    # noinspection PyTypeChecker
    builder = StateGraph(JournalState)  # no_qa

    builder.add_node(Node.GET_AI_RESPONSE, make_get_ai_response(llm=conversation_llm, session_store=session_store))
    builder.add_node(Node.RETRIEVE_HISTORY, make_retrieve_history(fragment_store=fragment_store))
    builder.add_node(Node.INTENT_CLASSIFIER, make_intent_classifier(llm=classifier_llm))
    builder.add_node(Node.PROFILE_SCANNER, make_profile_scanner(llm=classifier_llm, profile_store=profile_store))

    builder.add_node(Node.REFLECT,
                     make_reflect_node(reflection_graph, fragment_store=fragment_store, insights_store=insights_store))
    if claim_reflection_graph is not None and subjects_repo is not None:
        builder.add_node(Node.REFLECT2,
                         make_claim_reflect_node(claim_reflection_graph, subjects_repo=subjects_repo))
    builder.add_node(Node.RECALL, make_recall_node(fragment_store=fragment_store))
    _capture_store = capture_store or CaptureRepository(pg_gateway=fragment_store._pg)
    builder.add_node(Node.CAPTURE, make_capture_node(capture_store=_capture_store))
    builder.add_node(Node.CAPTURE_SEARCH, make_capture_search_node(capture_store=_capture_store))

    # START dispatches by user_command.
    start_targets = [Node.REFLECT, Node.RECALL, Node.CAPTURE, Node.CAPTURE_SEARCH, Node.INTENT_CLASSIFIER]
    if claim_reflection_graph is not None and subjects_repo is not None:
        start_targets.append(Node.REFLECT2)
    builder.add_conditional_edges(START, route_on_start, start_targets)

    # Intent / profile / retrieve all funnel into get_ai_response.
    builder.add_conditional_edges(Node.INTENT_CLASSIFIER, route_on_intent,
                                  [Node.GET_AI_RESPONSE, Node.PROFILE_SCANNER, Node.RETRIEVE_HISTORY, END])
    builder.add_conditional_edges(Node.PROFILE_SCANNER, route_on_profile,
                                  [Node.GET_AI_RESPONSE, Node.RETRIEVE_HISTORY, END])
    builder.add_conditional_edges(Node.RETRIEVE_HISTORY, goto(Node.GET_AI_RESPONSE),
                                  [Node.GET_AI_RESPONSE, END])

    # REFLECT and RECALL both finish with an AI response so the user sees
    # the recalled / reflected content rendered in conversation.
    builder.add_conditional_edges(Node.REFLECT, goto(Node.GET_AI_RESPONSE),
                                  [Node.GET_AI_RESPONSE, END])
    if claim_reflection_graph is not None and subjects_repo is not None:
        builder.add_conditional_edges(Node.REFLECT2, goto(Node.GET_AI_RESPONSE),
                                      [Node.GET_AI_RESPONSE, END])
    builder.add_conditional_edges(Node.RECALL, goto(Node.GET_AI_RESPONSE),
                                  [Node.GET_AI_RESPONSE, END])
    builder.add_conditional_edges(Node.CAPTURE_SEARCH, goto(Node.GET_AI_RESPONSE),
                                  [Node.GET_AI_RESPONSE, END])

    # CAPTURE just sets system_message; the runner reads it after the turn.
    builder.add_edge(Node.CAPTURE, END)

    # Final exit for the conversation path.
    builder.add_edge(Node.GET_AI_RESPONSE, END)

    return builder.compile(checkpointer=checkpointer)


def build_end_of_session_graph(
        registry: LLMRegistry,
        fragment_store: FragmentRepository,
        transcript_store: TranscriptRepository | None = None,
        thread_store: ThreadsRepository | None = None,
        classified_thread_store: ThreadsRepository | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the end-of-session classification + persistence pipeline.

    One node, one edge.  The pipeline is linear ETL with no branching — a
    7-node graph was the wrong shape for it.  All phases run sequentially
    inside ``end_of_session``; see ``eos_pipeline.make_end_of_session_node``
    for the phase sequence and error handling.

    Shares the JournalState schema and (optionally) the same checkpointer
    as the conversation graph, so it reads the final state the conversation
    left behind under the same ``thread_id``.
    """
    classifier_llm = registry.get("classifier")
    extractor_llm = registry.get("extractor")

    # noinspection PyTypeChecker
    builder = StateGraph(JournalState)  # no_qa

    builder.add_node(
        Node.END_OF_SESSION,
        make_end_of_session_node(
            transcript_store=transcript_store,
            thread_store=thread_store,
            classified_thread_store=classified_thread_store,
            fragment_store=fragment_store,
            classifier_llm=classifier_llm,
            extractor_llm=extractor_llm,
        ),
    )
    builder.add_edge(START, Node.END_OF_SESSION)
    builder.add_edge(Node.END_OF_SESSION, END)

    return builder.compile(checkpointer=checkpointer)
