"""checkpointer.py — Async Postgres checkpointer for LangGraph.

The checkpointer persists the full ``JournalState`` between graph super-steps,
keyed by ``thread_id`` (which we set to the session_id).

Lifecycle:
    The async context manager yields a configured AsyncPostgresSaver and tears
    down its connection pool on exit. The checkpointer creates its own async
    connection — separate from the sync PgGateway used by the data layer —
    because LangGraph's async checkpointer requires async psycopg.

Custom serde:
    LangGraph's default JsonPlusSerializer falls back to msgpack for unknown
    types and warns on each unregistered class — and will block them entirely
    in a future version. We pre-register the domain types this app stores in
    JournalState so roundtrips preserve concrete Pydantic instances (notably
    Exchange.human / Exchange.ai Turn fields the EOS pipeline depends on).

Usage:
    async with make_postgres_checkpointer(setup=True) as checkpointer:
        graph = build_conversation_graph(..., checkpointer=checkpointer)
        await graph.ainvoke(state, config={"configurable": {"thread_id": sid}})

The ``setup=True`` flag creates the checkpoint tables (idempotent). It is safe
to pass on every run during development; in production, run setup once at
deploy time and pass ``setup=False`` thereafter.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from journal_agent.configure.settings import get_settings


# Domain types that appear (transitively) on JournalState. Registering them
# silences the "Deserializing unregistered type ..." warnings AND keeps the
# checkpointer from dropping nested Pydantic fields on roundtrip.
_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    # journal_agent.model.session
    ("journal_agent.model.session", "UserCommandValue"),
    ("journal_agent.model.session", "StatusValue"),
    ("journal_agent.model.session", "Role"),
    ("journal_agent.model.session", "Turn"),
    ("journal_agent.model.session", "Tag"),
    ("journal_agent.model.session", "Exchange"),
    ("journal_agent.model.session", "ThreadSegment"),
    ("journal_agent.model.session", "Fragment"),
    ("journal_agent.model.session", "Domain"),
    ("journal_agent.model.session", "PromptKey"),
    ("journal_agent.model.session", "ContextSpecification"),
    ("journal_agent.model.session", "UserProfile"),
    ("journal_agent.model.session", "VerifierStatus"),
    ("journal_agent.model.session", "Insight"),
    ("journal_agent.model.session", "ScoreCard"),
    # journal_agent.model.insights — Phase 11
    ("journal_agent.model.insights", "SubjectSnapshot"),
    # journal_agent.graph.state
    ("journal_agent.graph.state", "WindowParams"),
]


def _make_serde() -> JsonPlusSerializer:
    """JsonPlusSerializer that knows about our domain types.

    Without ``allowed_msgpack_modules`` the serializer logs deprecation
    warnings on every checkpoint load and (in a future LangGraph release)
    will refuse to deserialize them entirely.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def make_postgres_checkpointer(
    setup: bool = False,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield an AsyncPostgresSaver bound to the configured Postgres URL.

    Args:
        setup: If True, create checkpoint tables on entry. Idempotent —
            safe to pass repeatedly. Disable in hot paths where the cost of
            a no-op DDL probe matters.
    """
    url = get_settings().postgres_url
    async with AsyncPostgresSaver.from_conn_string(url, serde=_make_serde()) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer
