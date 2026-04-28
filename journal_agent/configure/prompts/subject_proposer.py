"""subject_proposer.py — prompt for proposing a NEW tracked idea.

Invoked when the stance classifier returned no votes (or only weak ones)
against existing subjects. Decides whether the fragment introduces a new
theme worth tracking. The natural failure mode is subject explosion — the
prompt is heavily biased toward returning null.

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.insights import ProposerResponse

VERSION = "v1"

TEMPLATE = f"""\
A fragment was classified against the user's existing tracked ideas and either produced no votes or only weak ones. Your job is to decide whether the fragment introduces a NEW idea worth tracking.

You will receive a list of EXISTING SUBJECTS (each with label and current_claim) and one FRAGMENT.

CORE RULES

CREATE A SUBJECT ONLY IF ALL FOUR HOLD:

1. It captures something about the user's BELIEFS, VALUES, PATTERNS, IDENTITY, or RELATIONSHIPS — not a one-off event, mood, or task.

2. It is SPECIFIC ENOUGH to evaluate against future fragments. You should be able to imagine future journal entries either supporting OR contradicting it. Vague themes ("personal growth", "things on my mind") fail this test.

3. It is NOT ALREADY COVERED, even loosely, by an existing subject. If a similar subject exists, do not create — that subject can absorb refined evidence and update its claim text over time.

4. A reasonable observer would EXPECT THIS THEME TO RECUR in the user's writing. Will future fragments plausibly speak to this same idea? If you have to imagine unusual future content for the theme to come up again, fail the test.

BIAS TOWARD NOT CREATING.

Most fragments do not introduce new tracked ideas. If you are uncertain, return null. The cost of waiting is low: if the theme is real, it will reappear and the next fragment can trigger creation. The cost of over-creation is high: it pollutes the subject space with one-off observations that never accumulate evidence.

A fragment about being frustrated with a printer is NOT "stance on technology". A fragment about a single bad sleep is NOT "relationship with sleep". A fragment about enjoying one specific book is NOT "stance on reading". Be vigilant against this kind of pattern-fabrication.

WHEN YOU DO CREATE

- `label`: short, stable, concept-shaped (e.g., 'stance on solitude', 'relationship with father', 'tension between ambition and rest'). Not 'thoughts about X' or 'feelings about Y'.
- `description`: one sentence on what this tracks.
- `initial_claim`: current best phrasing of the user's position based on this single fragment. Keep it falsifiable — future fragments must be able to support OR contradict it.
- `initial_vote`: stance, strength, and reasoning for the first vote. Strength here can be modest (e.g., 0.6) — you're working from limited evidence.

INPUT SHAPE

The user message is JSON with:
    existing_subjects: list of {{ label, current_claim }}
    fragment: {{ dated_at, text }}

OUTPUT SHAPE

{_schema_block(ProposerResponse)}

Return {{ "new_subject": null }} when in doubt. Do not narrate.
"""
