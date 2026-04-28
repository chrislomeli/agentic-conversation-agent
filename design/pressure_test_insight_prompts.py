"""Pressure-test the three insight-pipeline prompts against Claude.

Standalone — does NOT touch the journal_agent graph or stores. Pure prompt
calibration check before committing to schema/integration work.

Run: uv run python design/pressure_test_insight_prompts.py
"""

from __future__ import annotations

import json
import os
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv("/Users/chrislomeli/Source/SECRETS/.env")

MODEL = "claude-sonnet-4-5"  # production "classifier" role uses sonnet
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Schemas (LLM-facing structured outputs) ──────────────────────────────

class StanceVote(BaseModel):
    subject_id: str
    stance: Literal["support", "contradict"]
    strength: float = Field(ge=0.0, le=1.0)
    reasoning: str


class StanceResponse(BaseModel):
    votes: list[StanceVote]


class InitialVote(BaseModel):
    stance: Literal["support", "contradict"]
    strength: float = Field(ge=0.0, le=1.0)
    reasoning: str


class NewSubject(BaseModel):
    label: str
    description: str
    initial_claim: str
    initial_vote: InitialVote


class ProposerResponse(BaseModel):
    new_subject: NewSubject | None


class RegeneratorResponse(BaseModel):
    action: Literal["no_change", "rewrite", "fork_suggested"]
    new_claim_text: str | None = None
    change_summary: str | None = None
    fork_reasoning: str | None = None


# ── Prompts (the things under test) ──────────────────────────────────────

STANCE_PROMPT = """You are analyzing a personal journal fragment to decide whether it provides evidence for or against any of the user's tracked ideas.

For each candidate subject, decide whether the fragment casts a vote:

- A vote is recorded ONLY when the fragment provides clear signal — supporting or contradicting the claim. Silence is the default. If the fragment does not bear on a claim, do not vote.
- A fragment may vote on multiple subjects, and may cast BOTH a support and a contradict vote on the same subject when it expresses genuine ambivalence.
- Strength = your confidence the vote is well-founded:
  0.9 = explicit and unambiguous
  0.6 = clearly implied
  0.3 = weak inference
  Do not return strengths below 0.3 — omit the vote instead.
- Reasoning must quote or paraphrase the specific passage that drives the vote.

Empty list is a valid and common answer."""

PROPOSER_PROMPT = """A fragment was classified against existing tracked ideas and either produced no votes or only weak ones. Decide whether it introduces a NEW idea worth tracking.

Create a subject ONLY if all four hold:

1. It captures something about the user's beliefs, values, patterns, identity, or relationships — NOT a one-off event or mood.
2. It is specific enough to evaluate against future fragments — you can imagine future fragments either supporting OR contradicting it.
3. It is not already covered, even loosely, by an existing subject.
4. A reasonable observer would expect this theme to recur in future writing.

Bias toward NOT creating. Most fragments do not introduce new tracked ideas. If uncertain, return null — if the theme is real, it will reappear and the next fragment can trigger creation."""

REGENERATOR_PROMPT = """A tracked idea has accumulated new evidence since its current phrasing was written. Decide whether the phrasing should be updated.

Decide:

1. Does the current phrasing still accurately describe what the evidence supports? If yes, leave it.
2. If not, rewrite to reflect the user's CURRENT state — not a historical average. Recent strong evidence dominates older weak evidence; older strong evidence still counts. Do NOT split the difference between divergent eras.
3. The new phrasing must remain falsifiable: future fragments should be able to support OR contradict it.
4. If the evidence has fractured into two genuinely distinct ideas, flag a fork — do not mash them into one claim."""


# ── Structured-output helper via tool-use ─────────────────────────────────

def call_structured(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """Force structured output by giving Claude a single tool to call."""
    tool = {
        "name": "respond",
        "description": "Return your structured response.",
        "input_schema": schema.model_json_schema(),
    }
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": "respond"},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return schema.model_validate(block.input)
    raise RuntimeError("No tool_use block in response")


# ── Test fixtures ─────────────────────────────────────────────────────────

CANDIDATES_GENERIC = [
    {"id": "subj_solitude", "current_claim": "user values quiet alone time as restorative"},
    {"id": "subj_creative", "current_claim": "user is drawn to writing as an outlet for unspoken thoughts"},
    {"id": "subj_career", "current_claim": "user is questioning whether their current career path fits"},
    {"id": "subj_health", "current_claim": "user prioritizes regular physical exercise"},
    {"id": "subj_mother", "current_claim": "user has unresolved tension in their relationship with their mother"},
]

EXISTING_SUBJECTS = [
    {"label": "stance on solitude", "current_claim": "user values quiet alone time as restorative"},
    {"label": "stance on creative pursuits", "current_claim": "user is drawn to writing as an outlet"},
    {"label": "stance on career direction", "current_claim": "user is questioning their current career path"},
    {"label": "relationship with partner", "current_claim": "user feels secure in their marriage"},
]


# ── Tests ────────────────────────────────────────────────────────────────

def test_1_mundane_fragment():
    """Stance classifier should produce empty (or near-empty) votes for a mundane fragment.
    Risk: over-eager voting on the gym mention -> health subject.
    """
    fragment = "Went to the gym today, did 30 min on the treadmill. Came home, made pasta for dinner, watched two episodes of that show everyone's been talking about. Tired now."
    user = json.dumps({
        "candidate_subjects": CANDIDATES_GENERIC,
        "fragment": {"dated_at": "2026-04-15", "text": fragment},
    }, indent=2)
    result = call_structured(STANCE_PROMPT, user, StanceResponse)
    return ("Test 1: mundane fragment", result, "expected: empty list, or at most one vote with strength <0.5 on health")


def test_2_clear_support():
    """Stance classifier should produce a clear support vote on a relevant fragment."""
    fragment = "I keep coming back to meditation. There's something about the discipline of just sitting that grounds me when nothing else does. I've been practicing 20 minutes every morning for the last two months and it's becoming the most important part of my day."
    candidates = [
        {"id": "subj_contemplative", "current_claim": "user is curious about meditation but inconsistent with practice"},
        {"id": "subj_solitude", "current_claim": "user values quiet alone time as restorative"},
        {"id": "subj_career", "current_claim": "user is questioning their current career path"},
    ]
    user = json.dumps({
        "candidate_subjects": candidates,
        "fragment": {"dated_at": "2026-04-15", "text": fragment},
    }, indent=2)
    result = call_structured(STANCE_PROMPT, user, StanceResponse)
    return ("Test 2: clear support", result, "expected: strong support (0.7+) on subj_contemplative; possibly weaker support on solitude")


def test_3_one_off_mood():
    """Subject proposer should return null for a one-off frustration."""
    fragment = "Frustrated with the printer today. Spent an hour trying to make it work. Eventually gave up and went to the coffee shop down the street to use theirs."
    user = json.dumps({
        "existing_subjects": EXISTING_SUBJECTS,
        "fragment": {"dated_at": "2026-04-15", "text": fragment},
    }, indent=2)
    result = call_structured(PROPOSER_PROMPT, user, ProposerResponse)
    return ("Test 3: one-off mood", result, "expected: new_subject = null. failure mode: proposing 'frustration with technology'")


def test_4_clear_new_theme():
    """Subject proposer should propose a real new theme."""
    fragment = "I've been re-reading my dad's letters from when I was in college. There's a tenderness there I never noticed at the time. I think I've been carrying a wrong story about him for twenty years — that he was distant — when actually he was trying, in his own awkward way, to stay close. It's making me reconsider a lot of things."
    user = json.dumps({
        "existing_subjects": EXISTING_SUBJECTS,
        "fragment": {"dated_at": "2026-04-15", "text": fragment},
    }, indent=2)
    result = call_structured(PROPOSER_PROMPT, user, ProposerResponse)
    return ("Test 4: clear new theme", result, "expected: new_subject proposed, label about relationship with father, initial_vote support")


def test_5_trajectory_regeneration():
    """Claim regenerator should reflect the trajectory, not average divergent eras."""
    subject = {
        "label": "stance on Buddhism",
        "current_claim": "user is drawn to Buddhist philosophy and finds it grounding",
        "claim_written_at": "2024-04-01",
    }
    votes = [
        {"dated_at": "2024-01-15", "stance": "support", "strength": 0.7, "reasoning": "user wrote about reading the Pali Canon and finding it 'unexpectedly moving'"},
        {"dated_at": "2024-02-08", "stance": "support", "strength": 0.8, "reasoning": "user described attending a sangha sit and feeling 'finally home'"},
        {"dated_at": "2024-03-12", "stance": "support", "strength": 0.6, "reasoning": "user quoted Thich Nhat Hanh approvingly in journal"},
        {"dated_at": "2024-09-20", "stance": "contradict", "strength": 0.7, "reasoning": "user expressed frustration with Buddhist teachers, felt the practice 'wasn't fitting' anymore"},
        {"dated_at": "2024-10-04", "stance": "contradict", "strength": 0.8, "reasoning": "user described turning back to Stoicism instead, said it felt 'more honest'"},
        {"dated_at": "2024-11-18", "stance": "contradict", "strength": 0.7, "reasoning": "user wrote 'I think I was looking for permission to disengage, not awakening'"},
    ]
    user = json.dumps({"subject": subject, "recent_votes_chronological": votes}, indent=2)
    result = call_structured(REGENERATOR_PROMPT, user, RegeneratorResponse)
    return ("Test 5: trajectory regeneration", result, "expected: action=rewrite, new claim reflects user has MOVED AWAY from Buddhism, NOT 'mixed feelings'")


# ── Runner ────────────────────────────────────────────────────────────────

def main():
    tests = [test_1_mundane_fragment, test_2_clear_support, test_3_one_off_mood,
             test_4_clear_new_theme, test_5_trajectory_regeneration]
    for t in tests:
        name, result, expected = t()
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        print(f"EXPECTED: {expected}")
        print(f"ACTUAL:")
        print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
