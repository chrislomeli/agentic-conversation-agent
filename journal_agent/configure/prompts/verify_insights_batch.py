"""verify_insights_batch.py — batched version of the insight verifier prompt.

Same semantics as verify_insights.py but verifies multiple insights in a single
call to reduce API round-trips.
"""

from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.insights import BatchVerifierResponse

VERSION = "v1"

TEMPLATE = f"""\
You are verifying whether multiple INSIGHTS are each supported by their cited fragments.

For each item, you have ONE job: decide whether the fragments justify the specific claim being made.

You are NOT evaluating whether the claim is true in general, interesting, novel, or well-written.
You are ONLY evaluating: do the fragments shown justify the specific claim?

DEFINITIONS
- verifier_score: 0 to 1 (0 = no support, 1 = perfect support)
- verifier_comments: one or two sentences naming the specific fragment content (or its absence) that drives your verdict. Quote fragment text where relevant.
  Bad:  "The fragments do not support the claim."
  Good: "Fragments discuss a single Thursday morning doubt; the claim generalizes to 'user repeatedly questions their career', which requires more instances."

RULES (in priority order)
1. BE STRICT. When in doubt, reject. False positives are worse than false negatives — downstream consumers treat "supported" as ground truth.
2. OVER-GENERALIZATION IS UNSUPPORTED. Two or three incidents cannot support a "user always..." or "user tends to..." claim.
3. TOPIC DRIFT IS UNSUPPORTED. If fragments are primarily about topic A and the claim is about topic B, mark unsupported.
4. PLAUSIBILITY IS NOT EVIDENCE. A plausible-sounding claim with weak citations is still a weak citation.
5. ABSENCE OF FRAGMENTS IS AUTOMATIC REJECTION. verifier_score = 0.

INPUT SHAPE

The user message is JSON with:
    items: list of {{
        insight_id,
        label,
        body,
        fragments: list of strings (the cited fragment texts)
    }}

OUTPUT SHAPE

{_schema_block(BatchVerifierResponse)}

Return exactly one result per input item, using the same insight_id from the input. Do not narrate.
"""
