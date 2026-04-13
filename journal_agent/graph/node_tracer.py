import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter

from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState,
)

logger = logging.getLogger(__name__)

# ── Graph builder ─────────────────────────────────────────────────────────────

def node_trace(node_name: str | None = None):
    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        name = node_name or func.__name__

        @wraps(func)
        def wrapper(state: JournalState) -> dict:
            start = perf_counter()
            session_id = state.get("session_id", "unknown")
            try:
                result = func(state)
                elapsed = perf_counter() - start
                status = result.get("status") if isinstance(result, dict) else None
                if status == STATUS_ERROR:
                    logger.warning(
                        "Node %s completed with error in %.3fs (session_id=%s, status=%s, error_message=%s)",
                        name,
                        elapsed,
                        session_id,
                        status,
                        result.get("error_message") if isinstance(result, dict) else None,
                    )
                else:
                    logger.info(
                        "Node %s completed in %.3fs (session_id=%s, status=%s)",
                        name,
                        elapsed,
                        session_id,
                        status,
                    )
                return result
            except Exception:
                elapsed = perf_counter() - start
                logger.exception(
                    "Node %s failed in %.3fs (session_id=%s)",
                    name,
                    elapsed,
                    session_id,
                )
                raise

        return wrapper
