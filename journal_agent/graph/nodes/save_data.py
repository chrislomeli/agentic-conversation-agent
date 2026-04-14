import logging
from collections.abc import Callable

from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState, STATUS_FRAGMENTS_SAVED, STATUS_EXCHANGES_SAVED, STATUS_TRANSCRIPT_SAVED,
)
from journal_agent.storage.storage import JsonStore

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


def make_save_fragments() -> Callable[..., dict]:
    @node_trace("fragment_extractor")
    def save_fragments(state: JournalState):
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

    return save_fragments




