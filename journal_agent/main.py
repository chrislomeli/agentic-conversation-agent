"""Entry point for the interactive journal agent."""
from uuid import uuid4

from langchain_core.messages import BaseMessage

from journal_agent.comms.llm_registry import build_llm_registry
from journal_agent.configure.config_builder import LLM_ROLE_CONFIG, configure_environment, models
from journal_agent.graph.graph import build_journal_graph
from journal_agent.graph.state import  JournalState
from journal_agent.model.session import ContextSpecification, Status, UserProfile
from journal_agent.storage.chroma_fragment_store import ChromaFragmentStore
from journal_agent.storage.exchange_store import TranscriptStore
from journal_agent.storage.profile_store import UserProfileStore


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

    # create stores — all satisfy their respective protocols
    # (SessionStore, FragmentStore, ProfileStore)
    session_store = TranscriptStore()
    fragment_store = ChromaFragmentStore()
    profile_store = UserProfileStore()

    # user profile
    user_profile = profile_store.load_profile()

    # get previously stored messages - this assumes we always save transcripts to a retrievable store  - will this always be the case?
    seed_context: list[BaseMessage] = session_store.retrieve_transcript()

    session_id = str(uuid4())  # or loaded from prior session
    initial_state = JournalState(
        session_id=session_id,
        recent_messages=seed_context,
        session_messages=[],
        transcript=[],
        threads=[],
        classified_threads=[],
        fragments=[],
        retrieved_history=[],
        context_specification=ContextSpecification(),  # nodes that need it run after intent_classifier sets it
        user_profile=user_profile,
        status=Status.IDLE,
        error_message=None,
    )
    graph = build_journal_graph(
        registry=registry,
        session_store=session_store,
        fragment_store=fragment_store,
        profile_store=profile_store,
    )
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
