"""Entry point for the interactive journal agent."""
from uuid import uuid4

from langchain_core.messages import BaseMessage

from journal_agent.comms.llm_registry import build_llm_registry
from journal_agent.configure.config_builder import LLM_ROLE_CONFIG, configure_environment, models
from journal_agent.graph.graph import build_journal_graph
from journal_agent.graph.state import JournalState
from journal_agent.model.session import ContextSpecification, Status
from journal_agent.repository import (
    TranscriptStore,
    PgFragmentRepository,
    JsonlGateway,
    get_pg_gateway,
    UserProfileRepository,
    ThreadsRepository,
    TranscriptRepository,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: STORE WIRING
# ═══════════════════════════════════════════════════════════════════════════════


def _build_stores():
    """Return (session_store, fragment_store, profile_store,
             transcript_store, thread_store, classified_thread_store).

    Every write path fans out to JSONL and Postgres.
    """
    pg = get_pg_gateway()

    transcript_store = TranscriptRepository(JsonlGateway("transcripts"), pg)
    thread_store = ThreadsRepository(JsonlGateway("threads"), pg)
    classified_thread_store = ThreadsRepository(JsonlGateway("classified_threads"), pg)
    fragment_store = PgFragmentRepository(pg_gateway=pg)
    profile_store = UserProfileRepository(JsonlGateway("user_profile"), pg)

    # TranscriptStore is the pure buffer; repository handles persistence + conversion.
    session_store = TranscriptStore(repository=transcript_store)

    return (
        session_store,
        fragment_store,
        profile_store,
        transcript_store,
        thread_store,
        classified_thread_store,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Configure dependencies, build the graph, and run one interactive session."""
    settings = configure_environment()

    registry = build_llm_registry(
        settings=settings,
        models=models,
        role_config=LLM_ROLE_CONFIG,
    )

    (
        session_store,
        fragment_store,
        profile_store,
        transcript_store,
        thread_store,
        classified_thread_store,
    ) = _build_stores()

    # user profile
    user_profile = profile_store.load_profile()

    # previously stored messages — assumes transcripts are retrievable
    seed_context: list[BaseMessage] = session_store.retrieve_transcript()

    session_id = str(uuid4())
    initial_state = JournalState(
        session_id=session_id,
        recent_messages=seed_context,
        session_messages=[],
        transcript=[],
        threads=[],
        classified_threads=[],
        fragments=[],
        retrieved_history=[],
        context_specification=ContextSpecification(),
        user_profile=user_profile,
        status=Status.IDLE,
        error_message=None,
    )
    graph = build_journal_graph(
        registry=registry,
        session_store=session_store,
        fragment_store=fragment_store,
        profile_store=profile_store,
        transcript_store=transcript_store,
        thread_store=thread_store,
        classified_thread_store=classified_thread_store,
    )
    try:
        graph.invoke(initial_state)

    except KeyboardInterrupt:
        session_store.store_cache(session_id)
        print("\nInterrupted. Session saved.")
    print("done")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
