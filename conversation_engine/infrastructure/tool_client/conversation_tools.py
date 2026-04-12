"""
Conversation tools — ToolSpec definitions for the converse agent loop.

These tools give the LLM agency over the conversation:
  - ask_human: surface a message to the human and collect their response
  - revalidate: re-run integrity checks on the knowledge graph
  - mark_complete: signal that the conversation goal has been met

Each tool is a ToolSpec with Pydantic I/O models and a handler factory.
Handlers are created at runtime with closures over the injected dependencies
(CallHuman, ConversationContext, etc.), keeping the specs themselves pure.
"""

from __future__ import annotations

from collections.abc import Callable

from commons.tool_client import ToolSpec
from pydantic import BaseModel, Field

# ── ask_human ─────────────────────────────────────────────────────────


class AskHumanInput(BaseModel):
    """Input for the ask_human tool."""

    message: str = Field(description="The message to present to the human.")
    options: list[str] | None = Field(
        default=None,
        description="Optional multiple-choice options for the human.",
    )


class AskHumanOutput(BaseModel):
    """Output from the ask_human tool."""

    response: str = Field(description="The human's response text.")
    skipped: bool = Field(
        default=False,
        description="True if the human chose to skip or timed out.",
    )


def make_ask_human_tool(human_callable) -> ToolSpec:
    """
    Create an ask_human ToolSpec bound to a CallHuman implementation.

    Parameters
    ----------
    human_callable : CallHuman
        The human surface to use (ConsoleHuman, MockHuman, etc.)
    """
    from commons.human import HumanRequest

    def handler(input_: AskHumanInput) -> AskHumanOutput:
        response = human_callable(
            HumanRequest(
                prompt=input_.message,
                options=input_.options,
            )
        )
        return AskHumanOutput(
            response=response.content,
            skipped=response.skipped,
        )

    return ToolSpec(
        name="ask_human",
        description=(
            "Present a message to the human and collect their response. "
            "Use this to explain findings, ask questions, request confirmation, "
            "or have a collaborative discussion about the knowledge graph."
        ),
        input_model=AskHumanInput,
        output_model=AskHumanOutput,
        handler=handler,
    )


# ── revalidate ────────────────────────────────────────────────────────


class RevalidateInput(BaseModel):
    """Input for the revalidate tool."""

    reason: str = Field(
        description="Why you are re-running validation (e.g. 'human reported changes').",
    )


class RevalidateOutput(BaseModel):
    """Output from the revalidate tool."""

    total_findings: int = Field(description="Total number of findings after re-validation.")
    open_findings: int = Field(description="Number of unresolved findings.")
    resolved_findings: int = Field(description="Number of resolved findings.")
    summary: str = Field(description="Brief summary of the validation result.")


def make_revalidate_tool(context, get_findings: Callable) -> ToolSpec:
    """
    Create a revalidate ToolSpec bound to a ConversationContext.

    Parameters
    ----------
    context : ConversationContext
        The domain context to validate against.
    get_findings : callable
        Returns the current findings list (closure over state).
    """

    def handler(input_: RevalidateInput) -> RevalidateOutput:
        prior = get_findings()
        result = context.validate(prior)
        open_f = [f for f in result.findings if not f.resolved]
        resolved_f = [f for f in result.findings if f.resolved]
        summary = context.format_finding_summary(open_f)
        return RevalidateOutput(
            total_findings=len(result.findings),
            open_findings=len(open_f),
            resolved_findings=len(resolved_f),
            summary=summary,
        )

    return ToolSpec(
        name="revalidate",
        description=(
            "Re-run integrity validation on the knowledge graph. "
            "Use this after the human reports they have made changes, "
            "or when you want to check the current state of findings."
        ),
        input_model=RevalidateInput,
        output_model=RevalidateOutput,
        handler=handler,
    )


# ── mark_complete ─────────────────────────────────────────────────────


class MarkCompleteInput(BaseModel):
    """Input for the mark_complete tool."""

    reason: str = Field(
        description="Why the conversation is complete (e.g. 'all findings resolved').",
    )


class MarkCompleteOutput(BaseModel):
    """Output from the mark_complete tool."""

    acknowledged: bool = Field(
        default=True,
        description="Always true — confirms the completion was recorded.",
    )


def make_mark_complete_tool() -> ToolSpec:
    """Create a mark_complete ToolSpec."""

    def handler(input_: MarkCompleteInput) -> MarkCompleteOutput:
        return MarkCompleteOutput(acknowledged=True)

    return ToolSpec(
        name="mark_complete",
        description=(
            "Signal that the conversation goal has been met and the session "
            "should end. Use this when all findings are resolved, or the human "
            "has indicated they are done, or no further progress can be made."
        ),
        input_model=MarkCompleteInput,
        output_model=MarkCompleteOutput,
        handler=handler,
    )
