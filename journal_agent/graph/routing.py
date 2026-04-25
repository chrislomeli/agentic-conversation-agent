import logging
from typing import Callable

# journal_agent/graph/routing.py
from collections.abc import Mapping
from typing import Any, Callable
from langgraph.graph import END


from journal_agent.model.session import StatusValue

logger = logging.getLogger(__name__)


def _route_base(state, *, next_node: str, on_completion: str = END) -> str:
    if state.status == StatusValue.ERROR:
        logger.warning("Routing to END (id=%s, error=%s)", state.session_id, state.error_message)
        return END
    if state.status == StatusValue.COMPLETED:
        return on_completion
    return next_node


def goto(node: str, on_completion: str = END) -> Callable:
    def _goto(state) -> str:
        return _route_base(state, next_node=node, on_completion=on_completion)

    return _goto