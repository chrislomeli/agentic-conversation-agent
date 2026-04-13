"""Entry point for the interactive journal agent."""

from uuid import uuid4

from journal_agent.comms.llm_registry import build_llm_registry
from journal_agent.configure.config_builder import LLM_ROLE_CONFIG, configure_environment, models
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.graph import build_journal_graph
from journal_agent.graph.state import STATUS_IDLE
from journal_agent.storage.api import SessionStore
from langchain_core.messages import BaseMessage, SystemMessage


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
    session_store = SessionStore()

    # seed_context includes system prompt and previously stored messages
    seed_context: list[BaseMessage] = [SystemMessage(get_prompt("conversation"))]
    stored_messages: list[BaseMessage] = session_store.retrieve_context()
    seed_context.extend(stored_messages or [])

    session_id = str(uuid4())  # or loaded from prior session
    initial_state = {
        "session_id": session_id,
        "seed_context": seed_context,
        "session_messages": [],
        "classified_exchanges": [],
        "status": STATUS_IDLE,
        "error_message": None,
    }
    graph = build_journal_graph(registry=registry, session_store=session_store)
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
