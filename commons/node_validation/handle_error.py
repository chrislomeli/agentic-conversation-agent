"""Generic error-handler node for LangGraph graphs."""

import logging

from commons.node_validation.result_schema import NodeResult

logger = logging.getLogger(__name__)


def handle_error(state: dict) -> dict:
    """Log the error and mark the run as failed.

    This node is generic — it reads ``node_result`` from state and
    marks ``status = "error"``.  Works with any graph that puts
    a :class:`NodeResult` in ``state["node_result"]``.
    """
    result: NodeResult | None = state.get("node_result")
    if result and not result.ok and result.error:
        logger.error(
            "Node error [%s]: %s",
            result.error.code,
            result.error.message,
        )
    return {"status": "error"}
