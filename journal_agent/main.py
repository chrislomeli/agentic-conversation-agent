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
from journal_agent.graph.reflection_graph import build_reflection_graph, build_claim_reflection_graph
from journal_agent.model.session import Role, UserCommandValue, UserProfile
from journal_agent.stores import (
    CaptureRepository,
    InsightsRepository,
    JsonlGateway,
    FragmentRepository,
    SubjectsRepository,
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


def _build_stores():
    """Return (pg, session_store, fragment_store, profile_store, insights_repo,
             transcript_store, thread_store, classified_thread_store).

    Every write path fans out to JSONL and Postgres.
    """
    pg = get_pg_gateway()

    transcript_store = TranscriptRepository(JsonlGateway("transcripts"), pg)
    thread_store = ThreadsRepository(JsonlGateway("threads"), pg)
    classified_thread_store = ThreadsRepository(JsonlGateway("classified_threads"), pg)
    fragment_store = FragmentRepository(pg_gateway=pg)
    profile_store = UserProfileRepository(JsonlGateway("user_profile"), pg)
    insights_repo = InsightsRepository(JsonlGateway("insights"), pg)

    # TranscriptStore is the pure buffer; stores handles persistence + conversion.
    session_store = TranscriptStore(repository=transcript_store)

    return (
        pg,
        session_store,
        fragment_store,
        profile_store,
        insights_repo,
        transcript_store,
        thread_store,
        classified_thread_store,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Configure dependencies and run an interactive terminal session."""
    settings = configure_environment()
    print("Using", {k: v.value for k, v in LLM_ROLE_CONFIG.items()})

    registry = build_llm_registry(
        settings=settings,
        models=models,
        role_config=LLM_ROLE_CONFIG,
    )

    (
        pg,
        session_store,
        fragment_store,
        profile_store,
        insights_repo,
        transcript_store,
        thread_store,
        classified_thread_store,
    ) = _build_stores()

    # user profile
    try:
        user_profile = profile_store.load_profile()
    except Exception:
        user_profile = UserProfile()
        profile_store.save_profile(user_profile)

    # previously stored messages — seeds the FIRST turn only; subsequent turns
    # load conversation state from the checkpointer.
    seed_context: list[BaseMessage] = exchanges_to_messages(session_store.retrieve_transcript() or [])

    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    reflection_graph = build_reflection_graph(
        registry=registry,
        insights_repo=insights_repo,
    )

    subjects_repo = SubjectsRepository(pg_gateway=pg)
    capture_store = CaptureRepository(pg_gateway=pg)
    claim_reflection = build_claim_reflection_graph(
        registry=registry,
        subjects_repo=subjects_repo,
    )

    async with make_postgres_checkpointer(setup=True) as checkpointer:
        conversation = build_conversation_graph(
            registry=registry,
            session_store=session_store,
            fragment_store=fragment_store,
            insights_store=insights_repo,
            profile_store=profile_store,
            reflection_graph=reflection_graph,
            capture_store=capture_store,
            claim_reflection_graph=claim_reflection,
            subjects_repo=subjects_repo,
            checkpointer=checkpointer,
        )
        eos = build_end_of_session_graph(
            registry=registry,
            fragment_store=fragment_store,
            transcript_store=transcript_store,
            thread_store=thread_store,
            classified_thread_store=classified_thread_store,
            checkpointer=checkpointer,
        )

        # Diagram only the conversation graph — the EOS pipeline is linear
        # ETL and doesn't need a picture.
        with open("graph.png", "wb") as f:
            f.write(conversation.get_graph().draw_mermaid_png())

        first_turn = True
        try:
            while True:
                user_input = await asyncio.to_thread(get_console_input)
                parsed = parse_user_input(user_input)

                if parsed.quit:
                    break

                turn_input = build_turn_input(parsed, session_id=session_id)
                if first_turn:
                    # First invocation for this thread_id has no prior state
                    # in the checkpointer — seed the bootstrap fields here.
                    turn_input["user_profile"] = user_profile
                    turn_input["recent_messages"] = seed_context
                    first_turn = False

                # Plain conversation turns get logged to the session buffer
                # so on_ai_turn can pair them into Exchanges.
                if parsed.command == UserCommandValue.NONE and parsed.message:
                    session_store.on_human_turn(
                        session_id=session_id,
                        role=Role.HUMAN,
                        content=parsed.message,
                    )

                events = conversation.astream_events(
                    turn_input, config=config, version="v2"
                )
                await stream_ai_response_to_terminal(events)

                # Surface any system_message the graph produced (e.g. /save
                # confirmations from the CAPTURE node).
                snapshot = await conversation.aget_state(config)
                if msg := snapshot.values.get("system_message"):
                    display_console_output(msg)

        except KeyboardInterrupt:
            session_store.store_cache(session_id)
            print("\nInterrupted. Session saved.")
            return
        except Exception as e:
            raise(e)
        # End-of-session pipeline. Reads the final conversation state from
        # the checkpointer (same thread_id, same JournalState schema).
        print("\nClosing session...")
        await eos.ainvoke({}, config=config)
    print("done")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    asyncio.run(main())
