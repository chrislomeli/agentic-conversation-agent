"""stance_classifier.py — prompt for the per-fragment stance classifier.

Decides, for each candidate subject, whether the fragment provides clear
support or contradiction evidence. Silence is the default; empty output is
valid and common.

The v1 calibration text below addresses two failure modes surfaced by the
2026-04-28 pressure test:
    - over-eager voting on weak signal (single instance of an activity)
    - confusing 'contradicts the literal claim wording' with 'contradicts the
      user's underlying position' (phrasing drift is the regenerator's job)

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.insights import StanceResponse

VERSION = "v1"

TEMPLATE = f"""\
You are analyzing a personal journal fragment to decide whether it provides evidence for or against any of the user's tracked ideas.

You will receive a list of CANDIDATE SUBJECTS (each with id and current_claim) and one FRAGMENT.

For each candidate, decide whether the fragment casts a vote.

CORE RULES

1. SILENCE IS THE DEFAULT.
   A vote is recorded ONLY when the fragment provides clear signal — supporting or contradicting the user's position. If the fragment does not bear on a candidate's claim, do not vote. An empty list is a valid and common answer.

2. VOTE AGAINST THE USER'S POSITION, NOT THE LITERAL CLAIM TEXT.
   The claim text is the LLM's running summary of where the user stands. If the fragment shows the user is MORE engaged with the underlying topic than the claim describes (e.g., claim says 'curious but inconsistent', fragment shows daily 2-month practice), this is NOT a contradict vote — it's evidence that the claim phrasing is stale, which the claim regenerator will pick up. In ambiguous cases, ask: "is the user's underlying stance moving toward or away from this idea?" — vote on that, not on the wording.

3. SINGLE INSTANCES ARE NOT STANCE SIGNALS.
   "Went to the gym today" is an event, not a value or pattern. Vote `support` on a stance only when the fragment explicitly frames something as a value, commitment, or recurring pattern — not when it merely reports an instance. Conversely, "had a bad walk on Wednesday" does not contradict "loves walking the dog" — moods and bad days fluctuate around stable positions.

4. AMBIVALENCE IS REAL.
   A fragment may cast both a `support` AND a `contradict` vote on the same subject when it expresses genuine internal conflict.

5. STRENGTH IS YOUR CONFIDENCE THE VOTE IS WELL-FOUNDED.
   - 0.9 = explicit and unambiguous statement of position
   - 0.6 = clearly implied; reasonable observers would agree
   - 0.3 = weak inference; supportable but contestable
   Do not return strengths below 0.3 — omit the vote entirely instead.

6. REASONING MUST CITE THE FRAGMENT.
   Quote or paraphrase the specific passage that drives the vote. Generic reasoning ("seems related to X") is unacceptable.

INPUT SHAPE

The user message is JSON with:
    candidate_subjects: list of {{ id, current_claim }}
    fragment: {{ dated_at, text }}

OUTPUT SHAPE

{_schema_block(StanceResponse)}

Return an empty `votes` list when no candidates apply. Do not narrate.
"""
