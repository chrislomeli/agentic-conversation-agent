import logging
from collections.abc import Callable

from typing_extensions import deprecated

from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState,
    STATUS_FRAGMENTS_SAVED,
    STATUS_EXCHANGES_SAVED,
    STATUS_TRANSCRIPT_SAVED,
    STATUS_THREADS_SAVED,
    STATUS_CLASSIFIED_THREADS_SAVED,
)
from journal_agent.storage.storage import JsonStore
from journal_agent.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

def make_save_transcript() -> Callable[..., dict]:
    @node_trace("save_transcript")
    def save_transcript(state: JournalState) -> dict:
        try:
            # store under session name
            session_id = state["session_id"]

            # content
            content = state["transcript"]

            # storage logic
            store = JsonStore("transcripts")

            # save exchanges
            store.save_session(session_id, content)

            return {
                "status": STATUS_TRANSCRIPT_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_transcript


@deprecated("Old one-shot pipeline — use make_save_threads + make_save_classified_threads.")
def make_save_exchanges() -> Callable[..., dict]:
    @node_trace("save_exchanges")
    def save_exchanges(state: JournalState) -> dict:
        try:
            # store under session name
            session_id = state["session_id"]

            # content
            content = state["classified_exchanges"]

            # storage logic
            store = JsonStore("exchanges")

            # save exchanges
            store.save_session(session_id, content)

            return {
                "status": STATUS_EXCHANGES_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_exchanges


def make_save_threads() -> Callable[..., dict]:
    @node_trace("save_threads")
    def save_threads(state: JournalState) -> dict:
        try:
            session_id = state["session_id"]
            content = state["threads"]

            store = JsonStore("threads")
            store.save_session(session_id, content)

            return {
                "status": STATUS_THREADS_SAVED
            }
        except Exception as e:
            logger.exception("Failed to save threads")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_threads


def make_save_classified_threads() -> Callable[..., dict]:
    @node_trace("save_classified_threads")
    def save_classified_threads(state: JournalState) -> dict:
        try:
            session_id = state["session_id"]
            content = state["classified_threads"]

            store = JsonStore("classified_threads")
            store.save_session(session_id, content)

            return {
                "status": STATUS_CLASSIFIED_THREADS_SAVED
            }
        except Exception as e:
            logger.exception("Failed to save classified threads")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_classified_threads


def make_save_fragments_to_json() -> Callable[..., dict]:
    @node_trace("save_fragments_to_json")
    def save_fragments_to_json(state: JournalState):
        try:
            # store under session name
            session_id = state["session_id"]

            # content
            content = state["fragments"]

            # storage logic
            store = JsonStore("fragments")

            # save exchanges
            store.save_session(session_id, content)

            return {
                "status": STATUS_FRAGMENTS_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_fragments_to_json


def make_save_fragments_to_vectordb(vector_store: VectorStore) -> Callable[..., dict]:
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
                "status": STATUS_FRAGMENTS_SAVED
            }

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return save_fragments_to_vectordb

