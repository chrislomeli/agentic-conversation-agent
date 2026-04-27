import asyncio
import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter

from journal_agent.graph.state import (
    JournalState, ReflectionState,
)
from journal_agent.model.session import StatusValue

logger = logging.getLogger(__name__)

# ── Node tracing decorator ───────────────────────────────────────────────────


def _log_result(name: str, elapsed: float, session_id: str, result: dict | None) -> None:
    """Shared logging for both sync and async wrappers."""
    status = result.get("status") if isinstance(result, dict) else None
    elapsed_ms = round(elapsed * 1000)
    if status == StatusValue.ERROR:
        logger.warning(
            "node completed with error",
            extra={
                "node": name,
                "session_id": session_id,
                "elapsed_ms": elapsed_ms,
                "status": str(status),
                "error_message": result.get("error_message") if isinstance(result, dict) else None,
            },
        )
    else:
        logger.info(
            "node completed",
            extra={
                "node": name,
                "session_id": session_id,
                "elapsed_ms": elapsed_ms,
                "status": str(status),
            },
        )


def node_trace(node_name: str | None = None):
    """Timing / logging decorator for LangGraph nodes.

    Wraps both sync and async node functions; records elapsed time,
    session_id, and status in structured ``extra`` fields so log
    aggregators can filter and group by node or session without parsing
    the message string.
    """
    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        name = node_name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(state: JournalState) -> dict:
                start = perf_counter()
                state_type = state.__class__.__name__
                if 'session_id' in state.model_fields:
                    session_id = state.session_id or f"<{state_type}>"
                else:
                    session_id = f"<{state_type}>"
                try:
                    result = await func(state)
                    _log_result(name, perf_counter() - start, session_id, result)
                    return result
                except Exception:
                    logger.exception(
                        "node raised exception",
                        extra={"node": name, "session_id": session_id,
                               "elapsed_ms": round((perf_counter() - start) * 1000)},
                    )
                    raise

            return async_wrapper

        @wraps(func)
        def wrapper(state: JournalState|ReflectionState) -> dict:
            start = perf_counter()
            state_type = state.__class__.__name__
            if 'session_id' in state.model_fields:
                session_id = state.session_id or f"<{state_type}>"
            else:
                session_id = f"<{state_type}>"

            try:
                result = func(state)
                _log_result(name, perf_counter() - start, session_id, result)
                return result
            except Exception:
                logger.exception(
                    "node raised exception",
                    extra={"node": name, "session_id": session_id,
                           "elapsed_ms": round((perf_counter() - start) * 1000)},
                )
                raise

        return wrapper

    return decorator
