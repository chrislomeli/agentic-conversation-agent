"""save_data.py — Persistence nodes for the end-of-session pipeline.

Each ``make_save_*`` factory returns a LangGraph node that writes one
pipeline artifact to storage and advances the status:

    save_transcript           → JsonStore("transcripts")
    save_threads              → JsonStore("threads")
    save_classified_threads   → JsonStore("classified_threads")
    save_fragments_to_json    → JsonStore("fragments")
    save_fragments_to_vectordb → VectorStore (ChromaDB)

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
from journal_agent.storage.storage import JsonStore
from journal_agent.storage.vector_store import VectorStore

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


def make_save_fragments_to_json(store: JsonStore | None = None) -> Callable[..., dict]:
    """Factory: persist extracted Fragments as JSONL."""
    store = store or JsonStore("fragments")

    @node_trace("save_fragments_to_json")
    def save_fragments_to_json(state: JournalState):
        try:
            session_id = state["session_id"]
            store.save_session(session_id, state["fragments"])

            return {
                "status": Status.FRAGMENTS_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return save_fragments_to_json


def make_save_fragments_to_vectordb(vector_store: VectorStore) -> Callable[..., dict]:
    """Factory: upsert Fragments into ChromaDB for semantic retrieval."""
    @node_trace("save_fragments_to_vectordb")
    def save_fragments_to_vectordb(state: JournalState):
        try:
            # store under session name
            session_id = state["session_id"]

            # content
            content = state["fragments"]


            # save exchanges
            vector_store.add_to_chroma_from_fragments(content)

            return {
                "status": Status.FRAGMENTS_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return save_fragments_to_vectordb

