"""score_card.py — Map LLM score cards to response strategies.

The intent classifier LLM returns a ScoreCard with three 0–1 floats
(question, first_person, task) plus per-domain scores. This module:

1. Thresholds each float into a boolean (> 0.5 = True).
2. Looks up the resulting (bool, bool, bool) triple in the Intent enum.
3. Maps the Intent to a ContextSpecification that tells ContextBuilder
   which prompt to use, how many messages to include, and whether to
   retrieve history from the vector store.

The 8 intents cover the full 2³ space:

    question  first_person  task  →  Intent
    --------  ------------  ----     ------
    T         T             T        SEEKING_HELP
    T         T             F        SELF_QUESTIONING
    T         F             T        RESEARCHING
    T         F             F        CURIOUS
    F         T             T        PLANNING
    F         T             F        MUSING
    F         F             T        DIRECTING
    F         F             F        OBSERVING

SEEKING_HELP and CURIOUS fall through to the default (conversational) spec.
SELF_QUESTIONING / MUSING / OBSERVING use the Socratic prompt.
RESEARCHING / DIRECTING / PLANNING use the Guidance prompt.
"""

from enum import Enum

from journal_agent.configure.prompts import PromptKey
from journal_agent.model.session import ScoreCard, ContextSpecification

THRESHOLDS = {
    "question":     0.5,
    "first_person": 0.5,
    "task":         0.5,
}


class Intent(Enum):
    SEEKING_HELP     = (True,  True,  True)
    SELF_QUESTIONING = (True,  True,  False)
    RESEARCHING      = (True,  False, True)
    CURIOUS          = (True,  False, False)
    PLANNING         = (False, True,  True)
    MUSING           = (False, True,  False)
    DIRECTING        = (False, False, True)
    OBSERVING        = (False, False, False)


# Built once at import time; ContextBuilder treats these as read-only.
_DEFAULT_SPEC = ContextSpecification(prompt_key=PromptKey.CONVERSATION)
_SOCRATIC_SPEC = ContextSpecification(prompt_key=PromptKey.SOCRATIC)
_GUIDANCE_SPEC = ContextSpecification(
    prompt_key=PromptKey.GUIDANCE,
    last_k_recent_messages=3,
    top_k_retrieved_history=0,
)

INTENT_TO_SPEC: dict[Intent, ContextSpecification] = {
    Intent.SELF_QUESTIONING: _SOCRATIC_SPEC,
    Intent.MUSING:           _SOCRATIC_SPEC,
    Intent.OBSERVING:        _SOCRATIC_SPEC,
    Intent.RESEARCHING:      _GUIDANCE_SPEC,
    Intent.DIRECTING:        _GUIDANCE_SPEC,
    Intent.PLANNING:         _GUIDANCE_SPEC,
    # SEEKING_HELP, CURIOUS → _DEFAULT_SPEC via .get() fallback below
}

def resolve_scorecard_to_specification(card: ScoreCard) -> ContextSpecification:
    """Map a ScoreCard to the ContextSpecification that will drive context assembly.

    Thresholds each dimension into a boolean triple, looks up the Intent,
    then returns the spec associated with that intent (or the default).
    """
    domains = [d.tag for d in card.domains if d.score  > 0.5]

    q  = card.question_score     > THRESHOLDS["question"]
    fp = card.first_person_score > THRESHOLDS["first_person"]
    t  = card.task_score         > THRESHOLDS["task"]
    intent = Intent((q, fp, t))
    return INTENT_TO_SPEC.get(intent, _DEFAULT_SPEC).model_copy(update={"tags": domains})




