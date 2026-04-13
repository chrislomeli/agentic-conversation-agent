"""
prompts.py — Named prompt templates for each agent role.

Add new entries to PROMPT_TEMPLATES when you introduce a new pipeline
stage that needs its own system prompt.  Nodes look up prompts by key
at graph-build time so the graph code never embeds raw prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_TEMPLATES: dict[str, str] = {
    # ── Conversation ──────────────────────────────────────────────────────
    "conversation": (
        "You are a thoughtful journal companion. "
        "Help the user explore their ideas. "
        "Always answer the question and, when relevant, note which broad "
        "subject area the conversation touches on."
    ),

    # ── Turn classifier ──────────────────────────────────────────────────
    "classifier": (
        "You are a classification engine. "
        "Given a list of conversation turns, label each turn with ONE or MORE primary "
        "subjects from the allowed taxonomy and up to THREE theme tags.\n\n"
        "Respond ONLY with valid JSON matching the schema provided."
    ),

    # ── Fragment extractor ───────────────────────────────────────────────
    "extractor": (
        "You are a knowledge extractor. "
        "Given a conversation turn and its classification, break it into "
        "atomic ideas (fragments). For each fragment provide:\n"
        "  • content  — the idea in one or two sentences\n"
        "  • summary  — a single-sentence headline\n"
        "  • tags     — 1-3 short tags\n\n"
        "Respond ONLY with valid JSON matching the schema provided."
    ),
}


# ── Taxonomies ────────────────────────────────────────────────────────────
# Predefined vocabularies that classifier prompts reference.
# Keep them here so they can be injected into prompts at call-time
# without hard-coding inside the prompt string itself.
@dataclass
class Ideation:
    key: str
    goals: str
    example: str
    notes: str
    theme_list: list[str]


TAXONOMY : list[Ideation] = [
    Ideation(
        key="creative_writing",
        goals="Assign this key to conversations that could be an interesting idea or kernel for creative writing - Use your imagination! ",
        example="We could be discussing the subject of human and AI interaction, even though the literal subject might be AI, it might be an interesting idea for exploration in a short story ",
        notes="Use the following themes if they fit, but add new ones if nothing fits",
        theme_list=["idea_seed", "concept"]
        ),
    Ideation(
        key="software_project",
        goals="Assign this key to conversations about projects that could be undertaken",
        example="We could be discussing a potential software project - we want to classify it as software_project and provide some sort of name",
        notes="There are no other themes other than a name for the project - if we did not discuss a name, feel free to make one up",
        theme_list=["<project name>"]
    ),
    Ideation(
        key="humanity",
        goals="Assign this key to conversations about philosophy AND human psychology",
        example="We could be discussing subjects related to the state of human existence, Stoicism, Buddhism, Cognitive Behavior ",
        notes="Use the following themes if they fit, but add your own if nothing fits",
        theme_list=["philosophy", "psychology"]
    )
]



SUBJECT_TAXONOMY: list[str] = [
    "creative_writing",
    "personal_goals",
    "project_ideas",
    "philosophy",
    "science",
    "technology",
    "health",
    "relationships",
    "career",
    "other",
]

THEME_TAXONOMY: list[str] = [
    "idea_seed",
    "decision",
    "question",
    "reflection",
    "plan",
    "emotion",
    "learning",
    "observation",
]


def get_prompt(key: str) -> str:
    """Return the prompt template for *key*, or raise KeyError."""
    try:
        return PROMPT_TEMPLATES[key]
    except KeyError:
        raise KeyError(
            f"Unknown prompt key {key!r}. "
            f"Available: {sorted(PROMPT_TEMPLATES)}"
        ) from None
