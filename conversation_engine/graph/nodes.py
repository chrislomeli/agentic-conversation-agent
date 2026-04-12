"""
Conversation graph node implementations.

Each function takes ConversationState and returns a partial state update.

Node responsibilities:
- preflight — run pre-flight LLM validation (once, on first turn)
- validate  — delegate to ConversationContext.validate(), produce Findings
- converse  — ReAct agent loop: LLM calls tools (ask_human, revalidate, etc.)

These nodes are **domain-agnostic**.  They work exclusively through the
ConversationContext protocol and the Finding type.  No domain-specific
imports (KnowledgeGraph, IntegrityRule, Assessment, etc.) appear here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conversation_engine.graph.context import ConversationContext, Finding
from conversation_engine.graph.state import ConversationState
from conversation_engine.infrastructure.llm import (
    CallLLM,
    LLMRequest,
    LLMResponse,
    LLMValidator,
    LLMValidatorReport,
    quiz_report_summary,
)
from conversation_engine.infrastructure.human import (
    CallHuman,
    HumanRequest,
    HumanResponse,
)
from conversation_engine.infrastructure.tool_client import (
    ToolClient,
    execute_tool_call,
    specs_to_langchain_tools,
)

logger = logging.getLogger(__name__)


# ── preflight ────────────────────────────────────────────────────────


def preflight(state: ConversationState) -> Dict[str, Any]:
    """
    Run pre-flight LLM validation on the first turn only.

    Pulls the quiz and system prompt from the ConversationContext.
    If no LLM or no quiz is configured, passes through immediately.
    On failure, sets status='error' so the router exits the graph.
    On success (or skip), sets preflight_passed=True so subsequent
    turns bypass this node.
    """
    # Skip if already passed (subsequent turns in the loop)
    if state.get("preflight_passed", False):
        return {}

    ctx: ConversationContext = state["context"]
    llm: Optional[CallLLM] = state.get("llm")
    quiz = ctx.preflight_quiz

    # Skip if no LLM or no quiz configured
    if not llm or not quiz:
        return {"preflight_passed": True}

    # Run the validator
    validator = LLMValidator(
        llm=llm,
        system_prompt=ctx.system_prompt,
        quiz=quiz,
        pass_threshold=0.7,
    )
    report = validator.run()

    if not report.passed:
        logger.error(
            "LLM pre-flight validation failed:\n%s",
            quiz_report_summary(report),
        )
        return {
            "status": "error",
            "preflight_passed": False,
            "messages": [
                AIMessage(
                    content=f"LLM failed pre-flight validation (score={report.weighted_score:.0%})"
                )
            ],
        }

    logger.info(
        "LLM pre-flight validation passed (score=%.0f%%)",
        report.weighted_score * 100,
    )
    return {"preflight_passed": True}


# ── validate ─────────────────────────────────────────────────────────


def validate(state: ConversationState) -> Dict[str, Any]:
    """
    Delegate validation to the injected ConversationContext.

    The context owns all domain logic (what to validate, how to
    evaluate rules, how to map violations to findings).  This node
    simply asks it to run and stores the result.
    """
    ctx: ConversationContext = state["context"]
    prior_findings = state.get("findings", [])

    result = ctx.validate(prior_findings)

    return {
        "findings": result.findings,
        "status": "running",
    }


# ── converse ─────────────────────────────────────────────────────────

MAX_AGENT_STEPS = 10  # safety limit on tool-call iterations per turn


def _build_findings_context(findings: List[Finding]) -> str:
    """Format open findings as text for the LLM prompt."""
    if not findings:
        return "No open findings — all integrity checks pass."
    lines = [f"Current findings ({len(findings)}):"]
    for f in findings:
        lines.append(f"  [{f.severity}] {f.finding_type}: {f.message}")
    return "\n".join(lines)


def _converse_simple(state: ConversationState) -> Dict[str, Any]:
    """
    Simple converse fallback — no tool_client, no agent loop.

    Used when no ToolClient is injected (backwards compat, basic tests).
    LLM produces text → optionally surface to human → bump turn.
    """
    ctx: ConversationContext = state["context"]
    open_findings = [f for f in state.get("findings", []) if not f.resolved]
    llm: Optional[CallLLM] = state.get("llm")
    human: Optional[CallHuman] = state.get("human")
    messages = state.get("messages", [])
    current_turn = state.get("current_turn", 0)

    if llm:
        findings_context = _build_findings_context(open_findings)
        history_lines = []
        for msg in messages:
            role = "AI" if msg.type == "ai" else "Human"
            history_lines.append(f"{role}: {msg.content}")
        history = "\n".join(history_lines)

        user_message = findings_context
        if history:
            user_message = (
                f"{findings_context}\n\n"
                f"Conversation so far:\n{history}\n\n"
                f"Continue the conversation. Respond to the human's "
                f"last message if they said something, otherwise "
                f"analyze the findings and suggest next steps."
            )

        request = LLMRequest(
            system_prompt=ctx.system_prompt,
            user_message=user_message,
            context={
                "session_id": state.get("session_id"),
                "current_turn": current_turn,
            },
        )

        try:
            response = llm(request)
            ai_text = response.content
        except Exception as e:
            ai_text = f"LLM error: {e}. {ctx.format_finding_summary(open_findings)}"
    else:
        ai_text = ctx.format_finding_summary(open_findings)

    new_messages: List = [AIMessage(content=ai_text)]

    if human:
        human_response = human(
            HumanRequest(
                prompt=ai_text,
                context={"turn": current_turn, "open_findings": len(open_findings)},
            )
        )
        if not human_response.skipped:
            new_messages.append(HumanMessage(content=human_response.content))

    return {
        "messages": new_messages,
        "current_turn": current_turn + 1,
    }


def _converse_agent(state: ConversationState) -> Dict[str, Any]:
    """
    ReAct agent loop — LLM decides which tools to call.

    The LLM has access to tools (ask_human, revalidate, mark_complete, etc.)
    and calls them as needed.  The loop continues until:
      - The LLM produces a plain text response (no tool calls)
      - The LLM calls mark_complete
      - MAX_AGENT_STEPS is reached (safety)

    Tool calls are executed via the ToolClient (validated, auditable).
    """
    ctx: ConversationContext = state["context"]
    open_findings = [f for f in state.get("findings", []) if not f.resolved]
    tool_client: ToolClient = state["tool_client"]
    current_turn = state.get("current_turn", 0)
    new_messages: List = []
    status_update: Optional[str] = None
    findings_update = None

    # Build the ChatOpenAI model with tools bound
    # We access the underlying ChatOpenAI from the adapter
    llm_callable: CallLLM = state["llm"]
    if hasattr(llm_callable, "_chat"):
        # OpenAICallLLM wraps a ChatOpenAI — use it directly
        chat_model = llm_callable._chat
    else:
        raise ValueError(
            "ReAct agent loop requires an OpenAI-compatible LLM adapter "
            "with a _chat attribute (e.g. OpenAICallLLM)."
        )

    # Get tool schemas from the client (uses list_tools() — the public interface)
    tool_schemas = specs_to_langchain_tools(tool_client)

    # todo - pros and cons for doing jit binding?
    llm_with_tools = chat_model.bind_tools(tool_schemas)

    # Build initial messages for the LLM
    from langchain_core.messages import SystemMessage

    findings_context = _build_findings_context(open_findings)

    chat_messages = [
        SystemMessage(content=ctx.system_prompt),
    ]

    # Include prior conversation history  - todo - not wrong or right but lets understand why we are not using using langchain macros to append to existing list
    prior_messages = state.get("messages", [])
    for msg in prior_messages:
        chat_messages.append(msg)

    # Add current findings as a new user message
    turn_prompt = (
        f"{findings_context}\n\n"
        f"You have tools available. Use ask_human to communicate with the human. "
        f"Use revalidate to re-check the knowledge graph after changes. "
        f"Use mark_complete when the conversation goal is met or no further progress can be made.\n\n"
        f"This is turn {current_turn + 1}. Analyze the findings and engage with the human."
    )
    chat_messages.append(HumanMessage(content=turn_prompt))

    # ── Agent loop ────────────────────────────────────────────────
    for step in range(MAX_AGENT_STEPS):
        logger.debug("Agent step %d/%d", step + 1, MAX_AGENT_STEPS)

        ai_response = llm_with_tools.invoke(chat_messages)
        chat_messages.append(ai_response)

        # If no tool calls, the LLM is done for this turn
        if not ai_response.tool_calls:
            if ai_response.content:
                new_messages.append(AIMessage(content=ai_response.content))
            break

        # Execute each tool call
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            logger.info("Agent calling tool: %s(%s)", tool_name, tool_call.get("args", {}))

            # Execute through the ToolClient (validated, auditable)
            result_text = execute_tool_call(tool_client, tool_call)

            # Feed result back to LLM as a ToolMessage
            chat_messages.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call["id"],
                )
            )

            # Handle special tool side-effects
            if tool_name == "mark_complete":
                status_update = "complete"
                logger.info("Agent called mark_complete — ending conversation")

            if tool_name == "revalidate":
                # Parse the result to update findings
                try:
                    result_data = json.loads(result_text)
                    logger.info(
                        "Revalidation: %d open, %d resolved",
                        result_data.get("open_findings", 0),
                        result_data.get("resolved_findings", 0),
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

            if tool_name == "ask_human":
                # Record the AI's message to the human
                ai_prompt = tool_call.get("args", {}).get("message", "")
                if ai_prompt:
                    new_messages.append(AIMessage(content=ai_prompt))

                # Parse human response and add to conversation messages
                try:
                    result_data = json.loads(result_text)
                    human_text = result_data.get("response", "")
                    skipped = result_data.get("skipped", False)
                    if human_text and not skipped:
                        new_messages.append(HumanMessage(content=human_text))
                except (json.JSONDecodeError, KeyError):
                    pass

        # If mark_complete was called, stop the loop
        if status_update == "complete":
            break
    else:
        logger.warning("Agent hit MAX_AGENT_STEPS (%d) — forcing turn end", MAX_AGENT_STEPS)

    # ── Build state update ────────────────────────────────────────
    update: Dict[str, Any] = {
        "messages": new_messages,
        "current_turn": current_turn + 1,
    }
    if status_update:
        update["status"] = status_update

    return update


def converse(state: ConversationState) -> Dict[str, Any]:
    """
    Collaborative AI/human exchange — dispatches to agent or simple mode.

    If a ToolClient is present in state, runs the ReAct agent loop
    where the LLM decides which tools to call (ask_human, revalidate, etc.).

    If no ToolClient, falls back to simple linear mode (LLM text → human → done).
    """
    tool_client = state.get("tool_client")
    llm = state.get("llm")

    if tool_client and llm:
        return _converse_agent(state)
    return _converse_simple(state)


# ── resolve_domain ──────────────────────────────────────────────────


def resolve_domain(state: ConversationState) -> Dict[str, Any]:
    """
    Resolve the DomainConfig into a ready-to-use ConversationContext.

    This is the first node in the graph.  It ensures that ``context``
    is populated before preflight / validate / converse run.

    Scenarios
    ---------
    1. ``context`` already set → pass through (caller built it themselves).
    2. ``project_name`` + ``project_store`` → load from store, build context.
    3. Neither → ask the human for a project name (via CallHuman), then
       return ``status="needs_project_name"`` so the router loops back.

    Returns a partial state update.
    """
    from conversation_engine.graph.architectural_context import (
        ArchitecturalOntologyContext,
    )

    # ── Scenario 1: context already provided ──────────────────────
    if state.get("context") is not None:
        logger.debug("resolve_domain: context already set — pass through")
        return {"status": "running"}

    # ── Scenario 2: load from store by project name ───────────────
    project_name = state.get("project_name")
    store = state.get("project_store")

    if project_name and store:
        config = store.load(project_name)
        if config is not None:
            ctx = ArchitecturalOntologyContext(config)
            logger.info("resolve_domain: loaded project '%s' from store", project_name)
            return {"context": ctx, "status": "running"}
        else:
            logger.warning("resolve_domain: project '%s' not found in store", project_name)
            return {
                "status": "error",
                "messages": [
                    AIMessage(content=f"Project '{project_name}' not found in the store.")
                ],
            }

    # ── Scenario 3: ask the human for a project name ─────────────
    human: Optional[CallHuman] = state.get("human")
    if human and not project_name:
        response = human(
            HumanRequest(
                prompt="Which project would you like to work on? Please provide the project name.",
                context={"reason": "no_project_name"},
                allow_skip=False,
            )
        )
        if response.content and not response.skipped:
            logger.info("resolve_domain: human provided project name '%s'", response.content)
            return {
                "project_name": response.content.strip(),
                "status": "needs_project_name",
            }

    # ── Fallback: cannot resolve ──────────────────────────────────
    logger.error("resolve_domain: no context, no project_name, and no human to ask")
    return {
        "status": "error",
        "messages": [
            AIMessage(content="Cannot start: no project name or domain configuration provided.")
        ],
    }
