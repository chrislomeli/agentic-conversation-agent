"""claim_regenerator.py — prompt for rewriting a stale claim.

Invoked when a subject has accumulated N new votes since the last claim
regeneration. The trigger is policy (config_builder.CLAIM_REGEN_VOTE_GAP).
The natural failure mode is averaging divergent eras into a hedged claim
that is true of nothing — the prompt explicitly forbids this and offers a
fork escape valve when the evidence has split into two distinct ideas.

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.insights import RegeneratorResponse

VERSION = "v1"

TEMPLATE = f"""\
A tracked idea has accumulated new evidence since its current phrasing was written. Your job is to decide whether the phrasing should be updated to reflect the evolved understanding.

You will receive:
    SUBJECT — { '{ label, current_claim, claim_written_at }' }
    RECENT VOTES — chronological list of { '{ dated_at, stance, strength, reasoning }' }

CORE RULES

1. IS THE CURRENT CLAIM STILL ACCURATE?
   If the claim still describes what the evidence supports, leave it alone (`action: no_change`). Don't rewrite for stylistic preference; only rewrite when the SUBSTANCE has shifted.

2. REWRITE TO REFLECT THE USER'S CURRENT STATE — NOT A HISTORICAL AVERAGE.
   If recent strong evidence diverges from older evidence, the new claim should reflect WHERE THE USER IS NOW, not split the difference. Example: ten supports of "drawn to Buddhism" followed by five strong contradicts ("turning back to Stoicism instead") should produce "user has moved away from Buddhist practice", NOT "user has mixed feelings about Buddhism". Hedged averaging is the failure mode to avoid.

3. RECENCY DOMINATES, BUT DOESN'T ERASE.
   Recent strong evidence dominates older weak evidence. Older STRONG evidence still counts as part of the trajectory — the user's history is part of the story. Acknowledge the evolution in `change_summary` rather than pretending the older era didn't happen.

4. FALSIFIABLE PHRASING.
   The new claim must remain something future fragments can support OR contradict. Avoid weasel words that make every future fragment ambiguous ("the user has a complicated relationship with X").

5. FORK ESCAPE VALVE.
   If the evidence has fractured into two genuinely DISTINCT ideas (not one shifting position, but two stable positions on related-but-separate things), do NOT mash them into one claim. Set `action: fork_suggested` and explain in `fork_reasoning`. Example: a subject "stance on Buddhism" whose votes have split into "user practices meditation regularly" and "user is skeptical of Buddhist metaphysics" — those are two stances, not one. Fork.

OUTPUT SHAPE

{_schema_block(RegeneratorResponse)}

Required fields by action:
    no_change       → action only; other fields null
    rewrite         → new_claim_text + change_summary required
    fork_suggested  → fork_reasoning required

Do not narrate.
"""
