#!/usr/bin/env python3
"""
config_builder.py — STEP 01: Environment setup & configuration

Load API keys from .env, configure logging, and enable LangSmith tracing.
This is the first thing every other step calls — get the environment ready
before touching the world engine or any agents.

The main() here is just a smoke-test: call configure_environment() and verify
the settings load without errors.
"""

import logging
import os
import warnings

from journal_agent.configure.settings import (
    LLM_ROLE_CONFIG,
    LLMModel,
    LLMLabel,
    LLMProvider,
    Settings,
    get_settings,
    models,
)

# ── Application defaults ──────────────────────────────────────────────────
# These constants are the starting values for ContextSpecification and
# UserProfile fields.  They live here (not in settings.py) because they
# are domain-level choices, not deployment/infra configuration.
FRAGMENTS_FETCH_LIMIT = 500
INSIGHTS_FETCH_LIMIT = 50
INSIGHTS_TOKEN_LIMIT = 1000
MINIMUM_CLUSTER_SCORE = 4.0          # cluster-level: size + 0.5*span_days
MINIMUM_CLUSTER_LABEL_SCORE = 0.5   # InsightDraft.confidence threshold
MINIMUM_VERIFIER_SCORE = 0.5

# ── Phase 11 — claim-based insights ──────────────────────────────────────
# Tunable knobs for the new insight pipeline. See design/phase11-claim-based-insights.md.
MIN_VOTE_STRENGTH = 0.3              # below this, the stance classifier should omit the vote
ROUTE_CANDIDATES_TOP_K = 8           # how many candidate subjects to surface per fragment
ROUTE_CANDIDATES_MIN_SIMILARITY = 0.5  # cosine similarity floor for candidate inclusion
CLAIM_REGEN_VOTE_GAP = 5             # regenerate claim text after this many new votes since last regen
SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH = 0.5  # if no existing-subject vote exceeds this, run the proposer
PROPOSER_DEDUP_SIMILARITY = 0.85     # in persist_votes, merge proposed subjects above this cosine similarity
DEFAULT_RECENT_MESSAGES_COUNT = 5
DEFAULT_SESSION_MESSAGES_COUNT = 10
DEFAULT_RETRIEVED_HISTORY_COUNT = 5
DEFAULT_RETRIEVED_HISTORY_DISTANCE = 3
DEFAULT_RESPONSE_STYLE="structured with headers"
DEFAULT_EXPLANATION_DEPTH="advanced: draw on academic and literary references naturally, the way an educated friend would in conversation, not as citations or lectures"
DEFAULT_TONE="warm and familiar, encouraging but not patronizing — a trusted friend who happens to be well-read. Speak as a peer, not an authority"
HUMAN_NAME="Chris"
AI_NAME=None
DEFAULT_LEARNING_STYLE="conceptual with examples: when exploring ideas, connect them to parallels in philosophy, literature, psychology, and history. Not to teach, but to enrich the conversation"
DEFAULT_INTERESTS=["philosophy", "musicality", "creative writing", "artificial intelligence", "software architecture"]
DEFAULT_PET_PEEVES= [
    "Never provide toy or simplified designs — always show professional-grade patterns",
    "Don't summarize or restate what I just said back to me — push the idea forward",
    "Don't be a yes-man — disagree when you have a reason to",
]


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _redacted_settings_json(settings: Settings) -> str:
    payload = settings.model_dump(mode="json")
    for key in ("anthropic_api_key", "openai_api_key", "langchain_api_key"):
        if key in payload:
            payload[key] = _mask_secret(payload.get(key, ""))
    llm_model = payload.get("llm_model")
    if isinstance(llm_model, dict) and "api_key" in llm_model:
        llm_model["api_key"] = _mask_secret(llm_model.get("api_key") or "")
    import json

    return json.dumps(payload, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
literals = {
    "AI_ENV_FILE": "/Users/chrislomeli/Source/SECRETS/.env",
    "USE_MODEL": LLMLabel.GPT_MINI,
}


def configure_environment() -> Settings:
    """
    Load API keys, configure logging, set up LangSmith tracing.

    Think of this as: "open the door, turn on the lights, get your tools ready"
    """
    os.environ.setdefault("AI_ENV_FILE", literals["AI_ENV_FILE"])
    settings = get_settings()  # Load .env file (API keys, project names, etc.)
    connection = models.get(literals["USE_MODEL"], models[LLMLabel.GPT_MINI])
    settings.llm_model = connection
    settings.llm_source = connection.provider if connection else LLMProvider.STUB

    # Set up Python logging so we can see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-35s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Suppress sklearn FutureWarning about HDBSCAN `copy` default change
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    # Route node_tracer output to a file so it never clutters the console chat
    node_tracer_logger = logging.getLogger("journal_agent.graph.node_tracer")
    node_tracer_logger.propagate = False
    _fh = logging.FileHandler("journal_agent.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(name)-35s  %(message)s", datefmt="%H:%M:%S"))
    node_tracer_logger.addHandler(_fh)

    # Print what we're using (redacted)
    print(_redacted_settings_json(settings))
    return settings
