"""seed_subject_from_cluster.py — propose a seed Subject from a cluster of related fragments.

Different bias from subject_proposer: a cluster is itself evidence of recurrence,
so creation is the default. Returning null is reserved for clusters that are
genuinely incoherent (mixed unrelated topics).

Used by the cold-start path in the claim reflection graph, which fires when
the active subject space is sparse (< COLD_START_SUBJECT_THRESHOLD subjects).

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

from journal_agent.configure.prompts.helpers import _schema_block
from journal_agent.model.insights import ProposerResponse

VERSION = "v1"

TEMPLATE = f"""\
You are seeding the user's tracked-subject space from a cluster of related personal-journal fragments. The clustering step already established that these fragments share a theme — your job is to name the theme as a stable Subject, write the user's current position as the initial Claim, and pick a representative initial vote.

INPUT SHAPE

The user message is JSON with:
    cluster: list of {{ fragment_id, dated_at, text }}

OUTPUT SHAPE

{_schema_block(ProposerResponse)}

DEFAULT TO CREATING.
The cluster is itself evidence of recurrence — the recurrence test is already passed. Return new_subject=null only when the cluster is genuinely incoherent (mixed unrelated topics that the clusterer should not have grouped).

WHEN YOU CREATE
- label: short, stable, concept-shaped (e.g., 'stance on solitude', 'relationship with father', 'tension between ambition and rest'). Not 'thoughts about X' or 'feelings about Y'.
- description: one sentence describing what this tracks.
- initial_claim: a falsifiable phrasing of the user's position that GENERALIZES across the cluster — not a paraphrase of one fragment. Future entries must be able to support OR contradict it.
- initial_vote: stance (support|contradict), strength in 0.5–0.7 (the cluster is collective evidence; individual fragments may not strongly state position), reasoning that names what the cluster collectively says.

Do not narrate. Return JSON matching the OUTPUT SHAPE.
"""
