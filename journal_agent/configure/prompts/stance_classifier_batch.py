"""stance_classifier_batch.py — batched version of the per-fragment stance classifier.

Same semantics as stance_classifier.py but accepts a list of items in a single
call, reducing API round-trips when there are many fragments to process.
"""

from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.insights import BatchStanceResponse

VERSION = "v1"

TEMPLATE = f"""\
You are analyzing multiple personal journal fragments to decide whether each provides evidence for or against the user's tracked ideas.

You will receive a list of ITEMS. Each item has a fragment_id, one FRAGMENT, and a list of CANDIDATE SUBJECTS.

For each item independently, decide which candidate subjects the fragment votes on.

CORE RULES

1. SILENCE IS THE DEFAULT.
   A vote is recorded ONLY when the fragment provides clear signal — supporting or contradicting the user's position. An empty votes list per item is valid and common.

2. VOTE AGAINST THE USER'S POSITION, NOT THE LITERAL CLAIM TEXT.
   If the fragment shows the user is MORE engaged than the claim describes, this is NOT a contradict vote — it signals the claim is stale, which the claim regenerator handles.

3. SINGLE INSTANCES ARE NOT STANCE SIGNALS.
   "Went to the gym today" is an event, not a pattern. Vote support only when the fragment explicitly frames something as a value, commitment, or recurring pattern.

4. AMBIVALENCE IS REAL.
   A fragment may cast both support AND contradict on the same subject.

5. STRENGTH IS YOUR CONFIDENCE THE VOTE IS WELL-FOUNDED.
   - 0.9 = explicit and unambiguous statement of position
   - 0.6 = clearly implied; reasonable observers would agree
   - 0.3 = weak inference; supportable but contestable
   Do not return strengths below 0.3 — omit the vote instead.

6. REASONING MUST CITE THE FRAGMENT.
   Quote or paraphrase the specific passage that drives the vote. Generic reasoning is unacceptable.

INPUT SHAPE

The user message is JSON with:
    items: list of {{
        fragment_id,
        fragment: {{ dated_at, text }},
        candidate_subjects: list of {{ id, current_claim }}
    }}

OUTPUT SHAPE

{_schema_block(BatchStanceResponse)}

Return exactly one result per input item, using the same fragment_id from the input.
Empty votes list is valid. Do not narrate.
"""
