"""
tool_client — Backwards-compatibility shim.

Core tool infrastructure has moved to ``commons.tool_client``.
Domain-specific tools (conversation_tools, project_graph_tools)
remain here because they depend on domain models.
"""

# ── Core (from commons) ─────────────────────────────────────────────
from commons.tool_client import (
    LocalToolClient,
    ToolCallError,
    ToolClient,
    ToolContentBlock,
    ToolRegistry,
    ToolResultEnvelope,
    ToolResultMeta,
    ToolSpec,
    execute_tool_call,
    specs_to_langchain_tools,
)

# ── Domain-specific tools (stay here) ───────────────────────────────
from conversation_engine.infrastructure.tool_client.conversation_tools import (
    AskHumanInput,
    AskHumanOutput,
    MarkCompleteInput,
    MarkCompleteOutput,
    RevalidateInput,
    RevalidateOutput,
    make_ask_human_tool,
    make_mark_complete_tool,
    make_revalidate_tool,
)
from conversation_engine.infrastructure.tool_client.project_graph_tools import (
    ProjectGraphInput,
    ProjectGraphOutput,
    make_project_graph_tool,
    make_project_spec_tool,
)

__all__ = [
    "ToolSpec",
    "ToolRegistry",
    "ToolContentBlock",
    "ToolResultEnvelope",
    "ToolResultMeta",
    "ToolClient",
    "ToolCallError",
    "LocalToolClient",
    "specs_to_langchain_tools",
    "execute_tool_call",
    # Domain-specific conversation tools
    "AskHumanInput",
    "AskHumanOutput",
    "RevalidateInput",
    "RevalidateOutput",
    "MarkCompleteInput",
    "MarkCompleteOutput",
    "make_ask_human_tool",
    "make_revalidate_tool",
    "make_mark_complete_tool",
    # Domain-specific project tools
    "ProjectGraphInput",
    "ProjectGraphOutput",
    "make_project_spec_tool",
    "make_project_graph_tool",
]
