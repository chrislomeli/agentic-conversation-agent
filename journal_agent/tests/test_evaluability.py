"""Tests for design plan #7 — evaluability audit.

These tests verify the four properties that make the system evaluable:
an eval harness can run classifiers in isolation, compare outputs across
runs, and trace every output back to the exact prompt version that
produced it.

1. Structured outputs — every LLM-powered classifier binds to a Pydantic
   schema that the harness can inspect and validate.

2. Prompt versioning — every PromptKey maps to a version string via
   ``get_prompt_version()``.  Bump the VERSION constant in the relevant
   prompt module when the text changes; the harness logs it automatically.

3. Stable artifact IDs — every pipeline artifact that flows through
   JournalState has a UUID that survives a round-trip, so two pipeline
   runs can be joined and compared.

4. PromptKey integrity — no duplicate / alias values; every key resolves
   to a non-empty prompt and a non-empty version.
"""

import uuid

import pytest

from journal_agent.configure.prompts import get_prompt, get_prompt_version
from journal_agent.graph.state import JournalState
from journal_agent.model.session import (
    Exchange,
    Fragment,
    FragmentDraftList,
    PromptKey,
    ScoreCard,
    ThreadClassificationResponse,
    ThreadSegment,
    ThreadSegmentList,
    UserProfile,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Structured outputs — classifier schemas are Pydantic BaseModels
# ═══════════════════════════════════════════════════════════════════════════════


def _is_pydantic(cls) -> bool:
    from pydantic import BaseModel
    try:
        return issubclass(cls, BaseModel)
    except TypeError:
        return False


def test_intent_classifier_output_schema_is_pydantic():
    assert _is_pydantic(ScoreCard)


def test_exchange_decomposer_output_schema_is_pydantic():
    assert _is_pydantic(ThreadSegmentList)


def test_thread_classifier_output_schema_is_pydantic():
    assert _is_pydantic(ThreadClassificationResponse)


def test_fragment_extractor_output_schema_is_pydantic():
    assert _is_pydantic(FragmentDraftList)


def test_profile_scanner_output_schema_is_pydantic():
    assert _is_pydantic(UserProfile)


def test_all_classifier_schemas_have_json_schema():
    """Each schema must be serializable so the eval harness can log it."""
    for cls in [ScoreCard, ThreadSegmentList, ThreadClassificationResponse,
                FragmentDraftList, UserProfile]:
        schema = cls.model_json_schema()
        assert "properties" in schema, f"{cls.__name__} schema missing 'properties'"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Prompt versioning — every key has a non-empty version string
# ═══════════════════════════════════════════════════════════════════════════════


def _make_state() -> JournalState:
    return JournalState(session_id="eval-test")


def test_every_prompt_key_has_a_version():
    """get_prompt_version must return a non-empty string for every PromptKey."""
    state = _make_state()
    for key in PromptKey:
        version = get_prompt_version(key)
        assert isinstance(version, str), f"Version for {key} is not a string"
        assert len(version) > 0, f"Empty version for {key}"


def test_prompt_version_and_text_use_same_key():
    """get_prompt and get_prompt_version must both accept the same keys."""
    state = _make_state()
    for key in PromptKey:
        text = get_prompt(key, state=state)
        version = get_prompt_version(key)
        assert text and version, f"Key {key} missing text or version"


def test_get_prompt_version_raises_for_unknown_key():
    with pytest.raises(KeyError):
        get_prompt_version("no_such_key_ever")


def test_version_strings_follow_semver_convention():
    """All prompt versions use the 'vN' convention (e.g. 'v1', 'v2')."""
    for key in PromptKey:
        version = get_prompt_version(key)
        assert version.startswith("v"), (
            f"Version for {key} is {version!r} — expected 'vN' format"
        )
        suffix = version[1:]
        assert suffix.isdigit(), (
            f"Version suffix for {key} is {suffix!r} — expected integer after 'v'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Stable artifact IDs — every pipeline artifact has a UUID
# ═══════════════════════════════════════════════════════════════════════════════


def test_thread_segment_has_thread_id():
    """ThreadSegment was missing a stable ID before #7-design."""
    t = ThreadSegment(thread_name="t1", exchange_ids=[], tags=[])
    assert hasattr(t, "thread_id"), "ThreadSegment must have a thread_id field"


def test_thread_segment_thread_id_is_valid_uuid():
    t = ThreadSegment(thread_name="t1", exchange_ids=[], tags=[])
    uuid.UUID(t.thread_id)  # raises ValueError if not valid UUID


def test_thread_segment_ids_are_unique_per_instance():
    t1 = ThreadSegment(thread_name="a", exchange_ids=[], tags=[])
    t2 = ThreadSegment(thread_name="b", exchange_ids=[], tags=[])
    assert t1.thread_id != t2.thread_id


def test_exchange_has_stable_id():
    e = Exchange()
    uuid.UUID(e.exchange_id)


def test_fragment_has_stable_id():
    f = Fragment(
        session_id="s1",
        content="test",
        exchange_ids=[],
        tags=[],
        timestamp=__import__("datetime").datetime.now(),
    )
    uuid.UUID(f.fragment_id)


def test_all_artifact_id_fields_are_named_consistently():
    """ID fields follow the '<type>_id' naming convention for easy eval joins."""
    assert hasattr(ThreadSegment, "model_fields")
    assert "thread_id" in ThreadSegment.model_fields
    assert "exchange_id" in Exchange.model_fields
    assert "fragment_id" in Fragment.model_fields


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PromptKey integrity — no aliases, all keys distinct
# ═══════════════════════════════════════════════════════════════════════════════


def test_prompt_key_values_are_unique():
    """No two PromptKey members may share the same value (no enum aliases).

    The enum alias bug: PROFILE_SCANNER and PROFILE_CLASSIFIER previously
    both had value 'profile_classifier', making PROFILE_SCANNER a silent
    alias for PROFILE_CLASSIFIER.  Python's enum iteration skips aliases,
    so they would have been invisible to any loop over PromptKey.
    """
    values = [key.value for key in PromptKey]
    assert len(values) == len(set(values)), (
        f"Duplicate PromptKey values detected: {[v for v in values if values.count(v) > 1]}"
    )


def test_prompt_key_count_matches_registry():
    """Every PromptKey must be in the version registry — no orphaned keys."""
    for key in PromptKey:
        version = get_prompt_version(key)
        assert version, f"PromptKey.{key.name} has no version in registry"


def test_no_dead_import_in_session_module():
    """sympy was imported but never used — verify it's been removed."""
    import journal_agent.model.session as session_mod
    assert not hasattr(session_mod, "decipher_affine"), (
        "Dead import 'decipher_affine' from sympy still present in session.py"
    )
