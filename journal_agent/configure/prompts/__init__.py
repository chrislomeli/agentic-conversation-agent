"""
prompts — named prompt templates for each agent role.

Each prompt lives in its own module. Most export a `TEMPLATE` string that is
fully resolved at import time (static prompts). A few export a
`PromptTemplate` with runtime placeholders (parametric prompts) — these are
formatted on each call with caller-supplied variables.

Look up prompts by key via `get_prompt(key, **vars)`. The enum `PromptKey`
is the source of truth for valid keys.
"""

from _helpers import TAXONOMY, taxonomy_json

from journal_agent.configure.prompts import (
    conversation,
    decomposer,
    exchange_classifier,
    extractor,
    guidance,
    intent_classifier,
    profile_scanner,
    socratic,
    thread_classifier,
)


__all__ = ["get_prompt", "taxonomy_json", "TAXONOMY"]

from base_prompt_template import PromptTemplateBuilder
from profile_scanner import UserProfileTemplate
from journal_agent.graph.state import JournalState
from journal_agent.model.session import PromptKey

_STATIC_REGISTRY: dict[str, str] = {
    PromptKey.INTENT_CLASSIFIER.value:   intent_classifier.TEMPLATE,
    PromptKey.CONVERSATION.value:        conversation.TEMPLATE,
    PromptKey.SOCRATIC.value:            socratic.TEMPLATE,
    PromptKey.GUIDANCE.value:            guidance.TEMPLATE,
    PromptKey.DECOMPOSER.value:          decomposer.TEMPLATE,
    PromptKey.THREAD_CLASSIFIER.value:   thread_classifier.TEMPLATE,
    PromptKey.EXCHANGE_CLASSIFIER.value: exchange_classifier.TEMPLATE,
    PromptKey.FRAGMENT_EXTRACTOR.value:  extractor.TEMPLATE,
}

_TEMPLATE_REGISTRY: dict[str, PromptTemplateBuilder] = {
    PromptKey.PROFILE_SCANNER.value: UserProfileTemplate(),
}


def get_prompt(key: str | PromptKey, state: JournalState) -> str:
    """Return the prompt for *key* as a fully-formatted string.

    Static prompts ignore ``prompt_vars``. Parametric prompts call
    ``PromptTemplate.format(**prompt_vars)`` and surface any missing-variable
    errors from LangChain.
    """
    lookup = key.value if isinstance(key, PromptKey) else key

    if lookup in _STATIC_REGISTRY:
        return _STATIC_REGISTRY[lookup]

    if lookup in _TEMPLATE_REGISTRY:
        return _TEMPLATE_REGISTRY[lookup].build(state)

    # available = sorted(set(_STATIC_REGISTRY) | set(_TEMPLATE_REGISTRY))
    raise KeyError(
        f"Unknown prompt key {lookup!r}."
    )



if __name__ == "__main__":
    from journal_agent.model.session import  UserProfile, ContextSpecification
    from journal_agent.model.session import  Status
    _state = JournalState(
        session_id="xyx",
        recent_messages=[],
        session_messages=[],
        transcript=[],
        threads=[],
        classified_threads=[],
        fragments=[],
        retrieved_history=[],
        context_specification=ContextSpecification(),  # nodes that need it run after intent_classifier sets it
        user_profile=UserProfile(),
        status=Status.IDLE,
        error_message=None,
    )

    prompt = get_prompt(PromptKey.PROFILE_SCANNER, _state)
    print(prompt)
