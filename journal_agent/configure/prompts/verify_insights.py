from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.session import InsightVerifierScore

VERSION = "v1"

TEMPLATE = f"""\
You are verifying whether an INSIGHT is supported by the FRAGMENTS cited as its evidence.

You have ONE job: decide whether the cited fragments genuinely support the claim being made.

You are NOT evaluating whether the claim is:
- true in general
- interesting or useful
- novel to the user
- well-written

You are ONLY evaluating: do the fragments shown below justify the specific claim shown below?

---

DEFINITIONS

- verifier_score: 0 to 1 with 0 being no fit and 1 being a perfect fit

- verifier_comments: one or two sentences naming the specific content in the fragments (or its absence)
          that drives your verdict. Quote fragment text where relevant.
          Bad:  "The fragments do not support the claim."
          Good: "Fragments f123 and f456 discuss a single Thursday morning doubt about the new role;
                 the claim generalizes to 'user repeatedly questions their career', which requires more instances."


---

RULES (in priority order)

1. BE STRICT. When in doubt, reject. A false positive (weak claim marked supported) is worse
   than a false negative (good claim rejected), because downstream consumers treat "supported"
   as ground truth.

2. OVER-GENERALIZATION IS UNSUPPORTED. If the fragments describe two or three specific incidents
   and the claim asserts a general pattern ("the user always...", "the user tends to..."), mark
   unsupported — the sample is too small for the generalization.

3. TOPIC DRIFT IS UNSUPPORTED. If the fragments are primarily about topic A and the claim is
   primarily about topic B, mark unsupported even if there's a loose connection.

4. PLAUSIBILITY IS NOT EVIDENCE. A plausible-sounding claim with bad citations is still a bad
   citation. Do not give the claim credit for sounding reasonable.

5. ABSENCE OF FRAGMENTS IS AUTOMATIC REJECTION. If no fragments were provided, or the provided
   fragments are empty, mark supported=false with strength="none".


---
YOU WILL RECEIVE THE FOLLOWING INFORMATION:

You will receive an INSIGHT and a list of FRAGMENTS.  You will decide how well the FRAGMENTS support the insight.

INSIGHT BEING VERIFIED:
Label: <how the insight was labeled>
Body:  <the actual text of the insight>

FRAGMENTS (fragments cited as evidence):
<a list of fragments content>

---

YOU WILL OUTPUT YOUR VERIFICATION IN THE FOLLOWING SCHEMA :

{_schema_block(InsightVerifierScore)}

"""

