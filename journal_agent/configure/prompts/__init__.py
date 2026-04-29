"""prompts — named prompt templates for each agent role.

Each prompt lives in its own module.  Static prompts export a ``TEMPLATE``
string resolved at import time.  Parametric prompts export a
``PromptTemplateBuilder`` subclass whose ``build(state)`` method renders
runtime variables.

Every module also exports a ``VERSION`` string (e.g. ``"v1"``).  Bump it
when the prompt text changes so eval runs can be traced back to a specific
prompt version.

Look up prompts by key via ``get_prompt(key, state)``
and versions via ``get_prompt_version(key)``.
``PromptKey`` is the source of truth for valid keys.
"""

from journal_agent.configure.prompts import (
    claim_regenerator,
    cluster_fragments,
    conversation,
    decomposer,
    exchange_classifier,
    extractor,
    guidance,
    intent_classifier,
    label_clusters,
    profile_scanner,
    seed_subject_from_cluster,
    socratic,
    stance_classifier,
    stance_classifier_batch,
    subject_proposer,
    thread_classifier,
    verify_insights,
    verify_insights_batch,
)

__all__ = ["get_prompt", "get_prompt_version"]

from .base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState
from journal_agent.model.session import PromptKey
from .conversation import ConversationProfileTemplate
from .socratic import SocraticProfileTemplate
from .guidance import GuidanceProfileTemplate
from .profile_scanner import UserProfileTemplate

_STATIC_REGISTRY: dict[str, str] = {
    PromptKey.INTENT_CLASSIFIER.value:   intent_classifier.TEMPLATE,
    PromptKey.DECOMPOSER.value:          decomposer.TEMPLATE,
    PromptKey.THREAD_CLASSIFIER.value:   thread_classifier.TEMPLATE,
    PromptKey.EXCHANGE_CLASSIFIER.value: exchange_classifier.TEMPLATE,
    PromptKey.FRAGMENT_EXTRACTOR.value:  extractor.TEMPLATE,
    PromptKey.VERIFY_INSIGHTS.value:     verify_insights.TEMPLATE,
    PromptKey.LABEL_CLUSTERS.value:      label_clusters.TEMPLATE,
    PromptKey.CREATE_CLUSTERS.value:     cluster_fragments.TEMPLATE,
    PromptKey.STANCE_CLASSIFIER.value:        stance_classifier.TEMPLATE,
    PromptKey.STANCE_CLASSIFIER_BATCH.value:  stance_classifier_batch.TEMPLATE,
    PromptKey.SUBJECT_PROPOSER.value:         subject_proposer.TEMPLATE,
    PromptKey.SEED_SUBJECT_FROM_CLUSTER.value: seed_subject_from_cluster.TEMPLATE,
    PromptKey.VERIFY_INSIGHTS_BATCH.value:    verify_insights_batch.TEMPLATE,
    PromptKey.CLAIM_REGENERATOR.value:        claim_regenerator.TEMPLATE,
}

_TEMPLATE_REGISTRY: dict[str, PromptTemplateBuilder] = {
    PromptKey.PROFILE_SCANNER.value: UserProfileTemplate(),
    PromptKey.CONVERSATION.value:    ConversationProfileTemplate(),
    PromptKey.SOCRATIC.value:        SocraticProfileTemplate(),
    PromptKey.GUIDANCE.value:        GuidanceProfileTemplate(),

}

_VERSION_REGISTRY: dict[str, str] = {
    PromptKey.INTENT_CLASSIFIER.value:   intent_classifier.VERSION,
    PromptKey.DECOMPOSER.value:          decomposer.VERSION,
    PromptKey.THREAD_CLASSIFIER.value:   thread_classifier.VERSION,
    PromptKey.EXCHANGE_CLASSIFIER.value: exchange_classifier.VERSION,
    PromptKey.FRAGMENT_EXTRACTOR.value:  extractor.VERSION,
    PromptKey.VERIFY_INSIGHTS.value:     verify_insights.VERSION,
    PromptKey.LABEL_CLUSTERS.value:      label_clusters.VERSION,
    PromptKey.CREATE_CLUSTERS.value:     cluster_fragments.VERSION,
    PromptKey.PROFILE_SCANNER.value:     profile_scanner.VERSION,
    PromptKey.CONVERSATION.value:        conversation.VERSION,
    PromptKey.SOCRATIC.value:            socratic.VERSION,
    PromptKey.GUIDANCE.value:            guidance.VERSION,
    PromptKey.STANCE_CLASSIFIER.value:        stance_classifier.VERSION,
    PromptKey.STANCE_CLASSIFIER_BATCH.value:  stance_classifier_batch.VERSION,
    PromptKey.SUBJECT_PROPOSER.value:         subject_proposer.VERSION,
    PromptKey.SEED_SUBJECT_FROM_CLUSTER.value: seed_subject_from_cluster.VERSION,
    PromptKey.VERIFY_INSIGHTS_BATCH.value:    verify_insights_batch.VERSION,
    PromptKey.CLAIM_REGENERATOR.value:        claim_regenerator.VERSION,
}


def get_prompt(key: str | PromptKey, state: JournalState | None = None) -> str:
    """Return the prompt for *key* as a fully-formatted string.

    Static prompts ignore ``state``.  Parametric prompts call
    ``PromptTemplateBuilder.build(state)`` and require ``state`` to be
    provided — raises ``ValueError`` if it's missing.
    """
    lookup = key.value if isinstance(key, PromptKey) else key

    if lookup in _STATIC_REGISTRY:
        return _STATIC_REGISTRY[lookup]

    if state is None:
        raise ValueError(f"State is required for parametric prompt {lookup!r}")

    if lookup in _TEMPLATE_REGISTRY:
        return _TEMPLATE_REGISTRY[lookup].build(state)

    raise KeyError(f"Unknown prompt key {lookup!r}")


def get_prompt_version(key: str | PromptKey) -> str:
    """Return the version string for *key* (e.g. ``"v1"``).

    Eval harnesses call this alongside ``get_prompt`` so they can record
    which prompt version produced a given model output.  Bump the
    ``VERSION`` constant in the relevant prompt module whenever the text
    changes in a semantically meaningful way.
    """
    lookup = key.value if isinstance(key, PromptKey) else key
    if lookup in _VERSION_REGISTRY:
        return _VERSION_REGISTRY[lookup]
    raise KeyError(f"Unknown prompt key {lookup!r}")
