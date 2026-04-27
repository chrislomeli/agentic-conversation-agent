"""FastAPI application for the journal agent.

Architecture
------------
FastAPI's ``lifespan`` context manager builds the full dependency graph once
at startup and tears it down on shutdown:

    Postgres checkpointer → LLM registry → stores → compiled LangGraph graphs

Endpoints
---------
POST   /sessions              — allocate a session_id; seeds bootstrap context
POST   /chat/{session_id}     — one conversation turn, streamed as SSE
DELETE /sessions/{session_id} — end the session, run the EOS pipeline
GET    /health                — liveness probe

Session bootstrap
-----------------
The first ``/chat`` call for a new session injects ``user_profile`` and
``recent_messages`` into the turn input — the same logic the terminal runner
applies on its first turn.  Subsequent calls read state from the checkpointer.

Run with:
    uv run uvicorn journal_agent.api.main:app --reload
"""
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from journal_agent.api.models import ChatRequest, SessionResponse, SseEvent
from journal_agent.api.streaming import format_sse, graph_stream
from journal_agent.telemetry import TelemetryCallbackHandler
from journal_agent.comms.commands import build_turn_input, parse_user_input
from journal_agent.comms.llm_registry import build_llm_registry
from journal_agent.configure.config_builder import (
    LLM_ROLE_CONFIG,
    configure_environment,
    models,
)
from journal_agent.graph.journal_graph import (
    build_conversation_graph,
    build_end_of_session_graph,
)
from journal_agent.graph.reflection_graph import build_reflection_graph
from journal_agent.model.session import Role, UserCommandValue, UserProfile
from journal_agent.stores import (
    InsightsRepository,
    JsonlGateway,
    FragmentRepository,
    ThreadsRepository,
    TranscriptRepository,
    TranscriptStore,
    UserProfileRepository,
    exchanges_to_messages,
    get_pg_gateway,
    make_postgres_checkpointer,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LIFESPAN — build dependencies once, share across requests
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the full dependency graph at startup; tear down on shutdown.

    The Postgres checkpointer holds an async connection pool that must stay
    open for the lifetime of the app — using it as an async context manager
    here ensures the pool is closed cleanly on SIGTERM.
    """
    settings = configure_environment()

    registry = build_llm_registry(settings=settings, models=models, role_config=LLM_ROLE_CONFIG)

    pg = get_pg_gateway()
    transcript_repo = TranscriptRepository(JsonlGateway("transcripts"), pg)
    thread_store = ThreadsRepository(JsonlGateway("threads"), pg)
    classified_thread_store = ThreadsRepository(JsonlGateway("classified_threads"), pg)
    fragment_store = FragmentRepository(pg_gateway=pg)
    profile_store = UserProfileRepository(JsonlGateway("user_profile"), pg)
    insights_repo = InsightsRepository(JsonlGateway("insights"), pg)

    # TranscriptStore is the in-session buffer; transcript_repo handles archiving.
    session_store = TranscriptStore(repository=transcript_repo)

    try:
        user_profile = profile_store.load_profile()
    except Exception:
        user_profile = UserProfile()
        # profile_store.save_profile(user_profile)  todo make sure we don't need to store first

    # Seed context is loaded once; it primes the FIRST turn of every session.
    seed_context = exchanges_to_messages(session_store.retrieve_transcript() or [])

    async with make_postgres_checkpointer(setup=True) as checkpointer:
        reflection_graph = build_reflection_graph(
            registry=registry,
            insights_repo=insights_repo,
        )
        conversation = build_conversation_graph(
            registry=registry,
            session_store=session_store,
            fragment_store=fragment_store,
            insights_store=insights_repo,
            profile_store=profile_store,
            reflection_graph=reflection_graph,
            checkpointer=checkpointer,
        )
        eos = build_end_of_session_graph(
            registry=registry,
            fragment_store=fragment_store,
            transcript_store=transcript_repo,
            thread_store=thread_store,
            classified_thread_store=classified_thread_store,
            checkpointer=checkpointer,
        )

        # Attach shared state to the app object for access in endpoint handlers.
        app.state.conversation = conversation
        app.state.eos = eos
        app.state.session_store = session_store
        app.state.user_profile = user_profile
        app.state.seed_context = seed_context
        # Set of session_ids that still need first-turn bootstrap injection.
        # Discarded once the first /chat call for that session is processed.
        app.state.new_sessions = set()

        yield
        # Checkpointer connection pool closes here on shutdown.


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: APP + ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="Journal Agent API",
    description="Streaming chat interface for the journal agent.",
    version="0.2.0",
    lifespan=lifespan,
)

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # prevents nginx from buffering the SSE stream
}


@app.post(
    "/sessions",
    summary="Create a new chat session",
    response_model=SessionResponse,
    status_code=201,
)
async def create_session() -> SessionResponse:
    """Allocate a new session_id.

    The client should call this before the first ``/chat`` message.  The
    returned ``session_id`` is used as a path parameter on all subsequent
    calls.  The first ``/chat`` call for this id will automatically inject
    the bootstrap context (user profile, recent message history).
    """
    session_id = str(uuid4())
    app.state.new_sessions.add(session_id)
    logger.info("session created", extra={"session_id": session_id})
    return SessionResponse(session_id=session_id)


@app.post(
    "/chat/{session_id}",
    summary="Send a message and stream the AI response",
    response_description="Server-sent events stream",
)
async def chat(session_id: str, request: ChatRequest) -> StreamingResponse:
    """One conversation turn for the given session.

    Parses the message for commands (``/reflect``, ``/recall``, ``/save``)
    and invokes the conversation graph for a single turn.  The AI response
    is streamed back as SSE events:

    - **token** — one chunk of AI text: ``{"text": "Hello "}``
    - **system** — graph feedback (e.g. ``/save`` confirmation): ``{"text": "..."}``
    - **done** — stream complete: ``{"text": ""}``
    - **error** — something went wrong: ``{"text": "error message"}``

    Sending ``/quit`` returns a ``system`` event advising the client to call
    ``DELETE /sessions/{session_id}`` to run the end-of-session pipeline.
    """
    logger.info("turn received", extra={"session_id": session_id, "message_len": len(request.message)})

    parsed = parse_user_input(request.message)

    if parsed.quit:
        async def _quit_stream():
            yield format_sse(SseEvent.SYSTEM, "Session ended. Call DELETE /sessions to save.")
            yield format_sse(SseEvent.DONE, "")

        return StreamingResponse(_quit_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)

    turn_input = build_turn_input(parsed, session_id=session_id)

    # Inject bootstrap fields on the first turn for this session.
    if session_id in app.state.new_sessions:
        turn_input["user_profile"] = app.state.user_profile
        turn_input["recent_messages"] = app.state.seed_context
        app.state.new_sessions.discard(session_id)

    # Log the human turn so the EOS pipeline can pair it with the AI turn
    # into an Exchange when the session ends.
    if parsed.command == UserCommandValue.NONE and parsed.message:
        app.state.session_store.on_human_turn(
            session_id=session_id,
            role=Role.HUMAN,
            content=parsed.message,
        )

    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [TelemetryCallbackHandler()],
    }
    events = app.state.conversation.astream_events(turn_input, config=config, version="v2")

    return StreamingResponse(
        graph_stream(events, app.state.conversation, config),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.delete(
    "/sessions/{session_id}",
    summary="End a session and run the end-of-session pipeline",
    status_code=202,
)
async def end_session(session_id: str) -> dict:
    """Trigger the EOS pipeline for the given session.

    Reads the final conversation state from the checkpointer (keyed by
    ``session_id`` as ``thread_id``), then runs the linear ETL pipeline:
    save transcript → decompose exchanges → save threads → classify threads
    → save classified → extract fragments → save fragments.

    Returns 202 once the pipeline completes (it runs synchronously before
    returning — the pipeline is fast relative to a user-initiated quit).
    """
    logger.info("eos triggered", extra={"session_id": session_id})
    app.state.new_sessions.discard(session_id)
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [TelemetryCallbackHandler()],
    }
    try:
        await app.state.eos.ainvoke({}, config=config)
    except Exception as exc:
        logger.error("eos pipeline failed", extra={"session_id": session_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info("eos completed", extra={"session_id": session_id})
    return {"status": "saved", "session_id": session_id}


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}
