"""
prompts.py — Named prompt templates for each agent role.

Add new entries to PROMPT_TEMPLATES when you introduce a new pipeline
stage that needs its own system prompt.  Nodes look up prompts by key
at graph-build time so the graph code never embeds raw prompt text.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from journal_agent.model.session import (
    ClassifiedExchange,
    Exchange,
    Fragment,
    Ideation,
)

# ── Taxonomy (loaded once from YAML) ──────────────────────────────────────
_TAXONOMY_PATH = Path(__file__).parent / "taxonomy.yaml"


def _load_taxonomy() -> list[Ideation]:
    with open(_TAXONOMY_PATH) as f:
        raw = yaml.safe_load(f)
    return [Ideation(**entry) for entry in raw]


TAXONOMY: list[Ideation] = _load_taxonomy()


def taxonomy_json() -> str:
    """Serialize the taxonomy for injection into prompts."""
    return json.dumps([asdict(t) for t in TAXONOMY], indent=2)


# ── Schema helpers ─────────────────────────────────────────────────────────
def _schema_block(model_cls: type) -> str:
    """Return a compact JSON-Schema representation for a Pydantic model."""
    return json.dumps(model_cls.model_json_schema(), indent=2)


# ── Prompt templates ───────────────────────────────────────────────────────
PROMPT_TEMPLATES: dict[str, str] = {
    # ── Conversation ──────────────────────────────────────────────────────
    "conversation": (
        "You are a thoughtful journal companion. "
        "Help the user explore their ideas. "
        "Always answer the question and, when relevant, note which broad "
        "subject area the conversation touches on."
    ),

    # ── Turn classifier ──────────────────────────────────────────────────
    "classifier": f"""
You are a classification engine.

You will receive a list of conversation exchanges serialized as JSON.
Each exchange conforms to this schema:

{_schema_block(Exchange)}

Your objective is to classify each exchange according to the Taxonomy
provided and produce a list of ClassifiedExchange objects.

ClassifiedExchange schema:

{_schema_block(ClassifiedExchange)}

Rules:
- Copy session_id and exchange_id from the input Exchange.
- human_summary: transcribe the relevant parts of the human message,
  or copy it verbatim if it does not need condensing.
- ai_summary: same treatment for the AI message.
- tags: assign one or more tags from the Taxonomy. Each tag object has
  a "tag" field (must be a value from the Taxonomy) and an optional
  "note" for justification.
- Read ALL exchanges together before classifying. Context matters —
  the human may refer to a previous exchange to hint at a category,
  e.g. "that last subject would be a good creative writing topic".
""",

    # ── Fragment extractor ───────────────────────────────────────────────
    "extractor": f"""
You are a knowledge extractor.

You will receive a list of ClassifiedExchange objects serialized as JSON.
ClassifiedExchange schema:

{_schema_block(ClassifiedExchange)}

Your objective is to produce Fragment records suitable for vector-database
embedding.  Output schema:

{_schema_block(Fragment)}

Rules:
- Fragments are NOT necessarily one-to-one with ClassifiedExchange records.
  Merge or split when it produces better standalone knowledge statements.
- The "content" field must be a dense, standalone knowledge statement —
  NOT a transcript summary.

  Bad:  "User asked if AI causes dependency. AI said AI would need to
         communicate for humans."
  Good: "AI-induced erosion of human communication would require AI
         systems capable of communicating on humans' behalf, making
         inter-AI communication a prerequisite."

  The good version reads like an idea, not a summary of a conversation.
  Embedders match queries like "does AI weaken human connection?" to
  prose that states the idea.
- Copy exchange_ids from all related ClassifiedExchange records.
- Copy tags from the related ClassifiedExchange records.
- Use the timestamp from the last related ClassifiedExchange, or the
  current datetime.
- Generate a unique UUID for the "id" field.
""",
}


def get_prompt(key: str) -> str:
    """Return the prompt template for *key*, or raise KeyError."""
    try:
        return PROMPT_TEMPLATES[key]
    except KeyError:
        raise KeyError(
            f"Unknown prompt key {key!r}. "
            f"Available: {sorted(PROMPT_TEMPLATES)}"
        ) from None
