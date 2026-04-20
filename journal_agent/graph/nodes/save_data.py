"""save_data.py — Persistence nodes for the end-of-session pipeline.

Each ``make_save_*`` factory returns a LangGraph node that writes one
pipeline artifact to storage and advances the status:

    save_transcript           → ArtifactStore (transcripts)
    save_threads              → ArtifactStore (threads)
    save_classified_threads   → ArtifactStore (classified_threads)
    save_fragments            → FragmentStore (JSONL + vector index in one call)

All nodes catch exceptions and return Status.ERROR so the graph can
route to END gracefully instead of crashing.
"""

import logging
from collections.abc import Callable

from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    JournalState,

)
from journal_agent.model.session import Status
from journal_agent.storage.protocols import ArtifactStore, FragmentStore
from journal_agent.storage.storage import JsonStore

logger = logging.getLogger(__name__)

def make_save_transcript(store: JsonStore | None = None) -> Callable[..., dict]:
    """Factory: persist the raw Exchange transcript as JSONL."""
    store = store or JsonStore("transcripts")

    @node_trace("save_transcript")
    def save_transcript(state: JournalState) -> dict:
        try:
            session_id = state["session_id"]
            store.save_session(session_id, state["transcript"])

            return {
                "status": Status.TRANSCRIPT_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return save_transcript


def make_save_threads(store: JsonStore | None = None) -> Callable[..., dict]:
    """Factory: persist decomposed ThreadSegments as JSONL."""
    store = store or JsonStore("threads")

    @node_trace("save_threads")
    def save_threads(state: JournalState) -> dict:
        try:
            session_id = state["session_id"]
            store.save_session(session_id, state["threads"])

            return {
                "status": Status.THREADS_SAVED
            }
        except Exception as e:
            logger.exception("Failed to save threads")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return save_threads


def make_save_classified_threads(store: JsonStore | None = None) -> Callable[..., dict]:
    """Factory: persist taxonomy-tagged ThreadSegments as JSONL."""
    store = store or JsonStore("classified_threads")

    @node_trace("save_classified_threads")
    def save_classified_threads(state: JournalState) -> dict:
        try:
            session_id = state["session_id"]
            store.save_session(session_id, state["classified_threads"])

            return {
                "status": Status.CLASSIFIED_THREADS_SAVED
            }
        except Exception as e:
            logger.exception("Failed to save classified threads")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return save_classified_threads


def make_save_fragments(fragment_store: FragmentStore) -> Callable[..., dict]:
    """Factory: persist Fragments via FragmentStore (handles both structured
    persistence and vector indexing in a single call)."""

    @node_trace("save_fragments")
    def save_fragments(state: JournalState):
        try:
            fragment_store.save_fragments(state["fragments"])

            return {
                "status": Status.FRAGMENTS_SAVED
            }

        except Exception as e:
            logger.exception("Failed to save fragments")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return save_fragments

