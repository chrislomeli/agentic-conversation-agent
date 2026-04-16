"""Entry point for the interactive journal agent."""

from uuid import uuid4

from langchain_core.messages import BaseMessage

from journal_agent.comms.llm_registry import build_llm_registry
from journal_agent.configure.config_builder import LLM_ROLE_CONFIG, configure_environment, models
from journal_agent.graph.graph import build_journal_graph
from journal_agent.graph.state import STATUS_IDLE
from journal_agent.storage.exchange_store import TranscriptStore
from journal_agent.storage.vector_store import get_vector_store


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Configure dependencies, build the graph, and run one interactive session."""
    # configuration and setup
    settings = configure_environment()

    registry = build_llm_registry(
        settings=settings,
        models=models,
        role_config=LLM_ROLE_CONFIG,
    )

    # create a session store
    # Data is saved under <project-root>/data/sessions by default.
    # Set JOURNAL_AGENT_ROOT to override the root directory.
    session_store = TranscriptStore()

    # create a vector store
    vector_store = get_vector_store()

    # get previously stored messages - this assumes we always save transcripts to a retrievable store  - will this always be the case?
    seed_context: list[BaseMessage] = session_store.retrieve_transcript()

    session_id = str(uuid4())  # or loaded from prior session
    initial_state = {
        "session_id": session_id,
        "seed_context": seed_context,
        "session_messages": [],
        "status": STATUS_IDLE,
        "error_message": None,
    }
    graph = build_journal_graph(registry=registry, session_store=session_store, vector_store=vector_store)
    try:
        graph.invoke(initial_state)

    except KeyboardInterrupt:
        # optional: flush pending turns
        session_store.store_cache(session_id)
        print("\nInterrupted. Session saved.")
    print("done")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
