"""Entry point for the interactive journal agent.

The terminal runner drives a Python loop around two compiled graphs:

    1. parse user input
    2. invoke the conversation graph for ONE turn
    3. consume token events to the terminal
    4. print any system_message the graph emitted (e.g. /save feedback)
    5. repeat

On ``/quit`` the runner breaks the loop and invokes the end-of-session graph
once against the same ``thread_id`` so it sees the final conversation state
the conversation graph left in the checkpointer.

The same shape works for the FastAPI endpoint — different transport, same
backend code path.
"""
import asyncio
from uuid import uuid4

from langchain_core.messages import BaseMessage

from journal_agent.comms.commands import build_turn_input, parse_user_input
from journal_agent.comms.human_chat import (
    get_console_input,
    stream_ai_response_to_terminal,
    display_console_output,
)
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: STORE WIRING
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Configure dependencies and run an interactive terminal session."""
    print("Using", {k: v.value for k, v in LLM_ROLE_CONFIG.items()})

    pg = get_pg_gateway()

    fragment_store = FragmentRepository(pg_gateway=pg)
    fragment_store.reembed_all()

    print("done")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
