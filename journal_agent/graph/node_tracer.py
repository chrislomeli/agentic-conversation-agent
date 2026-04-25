import asyncio
import logging
import os
from collections.abc import Callable
from functools import wraps
from time import perf_counter

from journal_agent.graph.state import (
    JournalState,
)
from journal_agent.model.session import StatusValue

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("NODE_TRACE_ENABLED", "false").lower() in ("1", "true", "yes")

# ── Node tracing decorator ───────────────────────────────────────────────────


def _log_result(name: str, elapsed: float, session_id: str, result: dict | None) -> None:
    """Shared logging for both sync and async wrappers."""
    status = result.get("status") if isinstance(result, dict) else None
    if status == StatusValue.ERROR:
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


def node_trace(node_name: str | None = None):
    """Timing / logging decorator for LangGraph nodes.

    Automatically detects whether the wrapped function is sync or async
    and produces the appropriate wrapper, so both ``def`` and ``async def``
    nodes can use the same ``@node_trace(...)`` decorator.

    Set NODE_TRACE_ENABLED=true to activate; disabled by default so output
    doesn't pollute the console chat interface (use LangSmith instead).
    """
    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        if not _ENABLED:
            return func
        name = node_name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(state: JournalState) -> dict:
                start = perf_counter()
                session_id = state.session_id
                try:
                    result = await func(state)
                    _log_result(name, perf_counter() - start, session_id, result)
                    return result
                except Exception:
                    logger.exception(
                        "Node %s failed in %.3fs (session_id=%s)",
                        name,
                        perf_counter() - start,
                        session_id,
                    )
                    raise

            return async_wrapper

        @wraps(func)
        def wrapper(state: JournalState) -> dict:
            start = perf_counter()
            session_id = state.session_id
            try:
                result = func(state)
                _log_result(name, perf_counter() - start, session_id, result)
                return result
            except Exception:
                logger.exception(
                    "Node %s failed in %.3fs (session_id=%s)",
                    name,
                    perf_counter() - start,
                    session_id,
                )
                raise

        return wrapper

    return decorator
