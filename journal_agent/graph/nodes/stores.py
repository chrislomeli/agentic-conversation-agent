"""stores.py — Persistence nodes for the end-of-session pipeline.

Each ``make_save_*`` factory returns a LangGraph node that writes one
pipeline artifact to stores and advances the status:

    save_transcript           → TranscriptRepository
    save_threads              → ThreadsRepository
    save_classified_threads   → ThreadsRepository
    save_fragments            → FragmentStore (vector index in one call)

All nodes catch exceptions and return Status.ERROR so the graph can
route to END gracefully instead of crashing.
"""

import logging
from collections.abc import Callable

from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import JournalState, ReflectionState
from journal_agent.model.session import StatusValue
from journal_agent.stores import FragmentRepository, TranscriptRepository, ThreadsRepository, InsightsRepository

logger = logging.getLogger(__name__)

def make_save_transcript(store: TranscriptRepository) -> Callable[..., dict]:
    """Factory: persist the raw Exchange transcript."""

    @node_trace("save_transcript")
    def save_transcript(state: JournalState) -> dict:
        try:
            session_id = state.session_id
            store.save_collection(session_id, state.transcript)

            return {
                "status": StatusValue.TRANSCRIPT_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return save_transcript


def make_save_threads(store: ThreadsRepository) -> Callable[..., dict]:
    """Factory: persist decomposed ThreadSegments."""

    @node_trace("save_threads")
    def save_threads(state: JournalState) -> dict:
        try:
            session_id = state.session_id
            store.save_collection(session_id, state.threads)

            return {
                "status": StatusValue.THREADS_SAVED
            }
        except Exception as e:
            logger.exception("Failed to save threads")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return save_threads


def make_save_classified_threads(store: ThreadsRepository) -> Callable[..., dict]:
    """Factory: persist taxonomy-tagged ThreadSegments."""

    @node_trace("save_classified_threads")
    def save_classified_threads(state: JournalState) -> dict:
        try:
            session_id = state.session_id
            store.save_collection(session_id, state.classified_threads)

            return {
                "status": StatusValue.CLASSIFIED_THREADS_SAVED
            }
        except Exception as e:
            logger.exception("Failed to save classified threads")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return save_classified_threads


def make_save_fragments(fragment_store: FragmentRepository) -> Callable[..., dict]:
    """Factory: persist Fragments via PgFragmentStore (handles both structured
    persistence and vector indexing in a single call)."""

    @node_trace("save_fragments")
    def save_fragments(state: JournalState):
        try:
            fragment_store.save_fragments(state.fragments)

            return {
                "status": StatusValue.FRAGMENTS_SAVED
            }

        except Exception as e:
            logger.exception("Failed to save fragments")
            return {
                "status": StatusValue.ERROR,
                "error_message": str(e),
            }

    return save_fragments


def make_save_insights(insights_repo: InsightsRepository) -> Callable[..., dict]:
    """Factory: embed + persist verified Insights via InsightsRepository."""

    @node_trace("save_insights")
    def save_insights(state: ReflectionState) -> dict:
        try:
            insights_repo.save_insights(state.verified_insights)
            return {"status": StatusValue.PROCESSING}
        except Exception as e:
            logger.exception("Failed to save insights")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return save_insights