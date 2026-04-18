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
    q  = card.question_score     > THRESHOLDS["question"]
    fp = card.first_person_score > THRESHOLDS["first_person"]
    t  = card.task_score         > THRESHOLDS["task"]
    intent = Intent((q, fp, t))
    return INTENT_TO_SPEC.get(intent, _DEFAULT_SPEC)




