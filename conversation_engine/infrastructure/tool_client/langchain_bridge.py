"""
LangChain bridge — convert ToolSpec definitions to LangChain tool format.

ChatOpenAI.bind_tools() accepts Pydantic models, dicts, or LangChain Tool
objects.  This bridge converts our ToolSpec input models into the format
that bind_tools() expects, keeping the ToolClient layer independent of
LangChain internals.

Usage:
    from conversation_engine.infrastructure.tool_client.langchain_bridge import (
        specs_to_langchain_tools,
        execute_tool_call,
    )

    # Bind tools to LLM
    tools = specs_to_langchain_tools(registry)
    llm_with_tools = chat_model.bind_tools(tools)

    # Execute a tool call from the LLM
    result = execute_tool_call(client, tool_call)
"""

from __future__ import annotations

from typing import Any

from conversation_engine.infrastructure.tool_client.client import ToolClient


def specs_to_langchain_tools(client: ToolClient) -> list[dict[str, Any]]:
    """
    Convert all tools in a ToolClient to OpenAI function-calling format.

    Uses client.list_tools() which returns MCP-style dicts with
    name/description/inputSchema.  Converts to the format
    ChatOpenAI.bind_tools() expects:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... JSON Schema ... }
            }
        }

    Returns a list of these dicts, one per registered tool.
    """
    catalog = client.list_tools()
    tools = []
    for entry in catalog:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": entry["name"],
                    "description": entry.get("description", ""),
                    "parameters": entry.get("inputSchema", {}),
                },
            }
        )
    return tools


def execute_tool_call(
    client: ToolClient,
    tool_call: dict[str, Any],
) -> str:
    """
    Execute a single tool call from the LLM response via the ToolClient.

    Parameters
    ----------
    client : ToolClient
        The tool client to execute through (validates I/O, wraps in envelope).
    tool_call : dict
        A LangChain AIMessage.tool_calls entry:
        {"name": "ask_human", "args": {"message": "..."}, "id": "call_xxx"}

    Returns
    -------
    str
        The text content of the tool result (for feeding back to the LLM
        as a ToolMessage).
    """
    name = tool_call["name"]
    args = tool_call.get("args", {})

    envelope = client.call(name, args)

    # Return structured JSON if available, otherwise text content
    if envelope.structured:
        import json

        return json.dumps(envelope.structured)

    if envelope.content:
        return envelope.content[0].text

    return "{}"
