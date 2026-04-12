"""Entry point for the interactive journal agent."""

from uuid import uuid4

from journal_agent.comms.llm_client import create_llm_client
from journal_agent.configure.config_builder import configure_environment
from journal_agent.graph.graph import build_journal_graph
from journal_agent.graph.state import STATUS_IDLE
from journal_agent.storage.api import SessionStore
from langchain_core.messages import BaseMessage, SystemMessage

JOURNAL_SYSTEM_PROMPT = (
    "You are a transcriber who classifies the content of our conversation into one of the following categories: astronomy, biology, chemistry, physics, or other.  "
    "Always provide the answer to the question and a classification for the question"
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Configure dependencies, build the graph, and run one interactive session."""
    # configuration and setup
    settings = configure_environment()
    model_config = settings.selected_model
    if model_config is None:
        raise ValueError("No LLM model is configured. Update USE_MODEL in configure/config_builder.py.")

    client = create_llm_client(
        provider=model_config.provider,
        api_key=model_config.api_key,
        model=model_config.model,
        base_url=settings.ollama_base_url,
    )

    # create a session store
    session_store = SessionStore()

    # seed_context includes system prompt and previously stored messages
    seed_context: list[BaseMessage] = [SystemMessage(JOURNAL_SYSTEM_PROMPT)]
    stored_messages: list[BaseMessage] = session_store.retrieve_context()
    seed_context.extend(stored_messages or [])

    session_id = str(uuid4())  # or loaded from prior session
    initial_state = {
        "session_id": session_id,
        "seed_context": seed_context,
        "session_messages": [],
        "status": STATUS_IDLE,
        "error_message": None,
    }
    graph = build_journal_graph(llm=client, session_store=session_store)
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
