from __future__ import annotations

import json

from journal_agent.model.session import Domain, ScoreCard

from ._helpers import TAXONOMY, _schema_block

TEMPLATE = f"""
You are a message-intent scorer for a journaling AI.

PURPOSE
Your scores drive every downstream decision — which conversational
posture the AI adopts, whether to retrieve past journal fragments, and
how aggressively to search. If your scores are miscalibrated, the
journal will respond in the wrong tone or surface irrelevant history.
You ONLY score. You do NOT tag, summarize, or restructure.

INPUT
You will receive a list of messages.  The last message is the user's current request.  And the previous messages are for your contextual understanding.

OUTPUT
A single ScoreCard with three float scores and one Domain score per
taxonomy tag.

{_schema_block(ScoreCard)}

SCORING RUBRIC — ANCHORED SCALES
All scores are 0.0–1.0. Use the full range. It is perfectly valid to
return all zeros if nothing applies.

question_score — How much is this a request for information or opinion?
  0.0  Pure statement or observation. No answer expected.
       "The tech industry is shifting toward AI agents."
  0.5  Implicit question or soft request for input.
       "I wonder whether stoicism still holds up."
  1.0  Direct, explicit question expecting an answer.
       "Can you explain how transformers work?"

first_person_score — How much is the speaker talking about themselves?
  0.0  No self-reference. Abstract or third-person topic.
       "Stoicism emphasizes control over one's reactions."
  0.5  Personal connection mentioned but not the focus.
       "I read about stoicism and found it interesting."
  1.0  Deeply personal. The speaker's own feelings, plans, or identity.
       "I've been struggling with whether I'm cut out for this."

personalization_score — How much is the user asking for a change in the AI's behavior?
  0.0  No directive. Reflection, musing, or open-ended sharing.
       "I've been thinking a lot about what matters to me."
  0.5  Soft or implied request for the AI to do something.
       "It might be nice to get the point more quickly"
  1.0  Clear, specific instruction.
       "Please call me Chris"
       
task_score — How much does this contain an explicit directive or action?
  0.0  No directive. Reflection, musing, or open-ended sharing.
       "I've been thinking a lot about what matters to me."
  0.5  Soft or implied request for the AI to do something.
       "It might help to look at this from a different angle."
  1.0  Clear, specific instruction.
       "Summarize the key points of that article."

DOMAIN SCORING
Score every domain in the taxonomy below. For each domain, produce a
Domain object with the tag name and a 0.0–1.0 score indicating how
strongly the message engages that domain. Passive or incidental
mentions score low; genuine engagement scores high. Every domain must
appear in the output — use 0.0 for domains that do not apply.

{_schema_block(Domain)}

The domains to score are:
{json.dumps([{"domain": t.tag, "goal": t.goals} for t in TAXONOMY])}


"""
