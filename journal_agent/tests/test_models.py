"""Layer 1 tests — Pydantic models in model/session.py."""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from journal_agent.configure.config_builder import (
    AI_NAME,
    DEFAULT_EXPLANATION_DEPTH,
    DEFAULT_INTERESTS,
    DEFAULT_LEARNING_STYLE,
    DEFAULT_PET_PEEVES,
    DEFAULT_RECENT_MESSAGES_COUNT,
    DEFAULT_RESPONSE_STYLE,
    DEFAULT_RETRIEVED_HISTORY_COUNT,
    DEFAULT_RETRIEVED_HISTORY_DISTANCE,
    DEFAULT_SESSION_MESSAGES_COUNT,
    DEFAULT_TONE,
    HUMAN_NAME,
)
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.state import JournalState
from journal_agent.model.session import (
    ClassifiedExchange,
    ContextSpecification,
    Domain,
    Exchange,
    Fragment,
    PromptKey,
    Role,
    ScoreCard,
    StatusValue,
    Tag,
    Turn,
    UserProfile,
)


def _make_state(**overrides) -> dict:
    """Minimal JournalState-compatible dict for prompt tests."""
    state = {
        "session_id": "test-session",
        "recent_messages": [],
        "session_messages": [],
        "transcript": [],
        "threads": [],
        "classified_threads": [],
        "fragments": [],
        "retrieved_history": [],
        "context_specification": ContextSpecification(),
        "user_profile": UserProfile(),
        "status": StatusValue.IDLE,
        "error_message": None,
    }
    state.update(overrides)
    return state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fragment(session_id: str = "s1") -> Fragment:
    return Fragment(
        session_id=session_id,
        content="some thought",
        exchange_ids=["e1"],
        tags=[Tag(tag="philosophy")],
        timestamp=datetime.now(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UserProfile
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserProfileDefaults:
    def test_instantiates_with_no_args(self):
        profile = UserProfile()
        assert isinstance(profile, UserProfile)

    def test_string_defaults_match_config_constants(self):
        p = UserProfile()
        assert p.response_style == DEFAULT_RESPONSE_STYLE
        assert p.explanation_depth == DEFAULT_EXPLANATION_DEPTH
        assert p.tone == DEFAULT_TONE
        assert p.learning_style == DEFAULT_LEARNING_STYLE
        assert p.human_label == HUMAN_NAME
        assert p.ai_label == AI_NAME

    def test_list_defaults_match_config_constants(self):
        p = UserProfile()
        assert p.interests == list(DEFAULT_INTERESTS)
        assert p.pet_peeves == list(DEFAULT_PET_PEEVES)

    def test_optional_fields_have_correct_none_defaults(self):
        p = UserProfile()
        assert p.decision_style is None
        assert p.ai_label is None

    def test_empty_list_defaults(self):
        p = UserProfile()
        assert p.active_projects == []
        assert p.recurring_themes == []

    def test_updated_at_defaults_to_now(self):
        before = datetime.now()
        p = UserProfile()
        after = datetime.now()
        assert before <= p.updated_at <= after


class TestUserProfileMutableDefaultIsolation:
    def test_interests_are_independent_between_instances(self):
        p1 = UserProfile()
        p2 = UserProfile()
        p1.interests.append("extra")
        assert "extra" not in p2.interests

    def test_pet_peeves_are_independent_between_instances(self):
        p1 = UserProfile()
        p2 = UserProfile()
        p1.pet_peeves.clear()
        assert len(p2.pet_peeves) == len(DEFAULT_PET_PEEVES)

    def test_active_projects_are_independent_between_instances(self):
        p1 = UserProfile()
        p2 = UserProfile()
        p1.active_projects.append("project-x")
        assert p2.active_projects == []


class TestUserProfileSerialization:
    def test_round_trips_through_json(self):
        original = UserProfile(human_label="Alice")
        json_str = original.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        assert restored.human_label == "Alice"
        assert restored.interests == original.interests
        assert restored.pet_peeves == original.pet_peeves

    def test_updated_at_survives_round_trip(self):
        original = UserProfile()
        restored = UserProfile.model_validate_json(original.model_dump_json())
        assert restored.updated_at == original.updated_at


# ═══════════════════════════════════════════════════════════════════════════════
# ContextSpecification
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextSpecificationDefaults:
    def test_defaults_match_config_constants(self):
        spec = ContextSpecification()
        assert spec.last_k_session_messages == DEFAULT_RECENT_MESSAGES_COUNT
        assert spec.last_k_recent_messages == DEFAULT_SESSION_MESSAGES_COUNT
        assert spec.top_k_retrieved_history == DEFAULT_RETRIEVED_HISTORY_COUNT
        assert spec.distance_retrieved_history == DEFAULT_RETRIEVED_HISTORY_DISTANCE

    def test_default_prompt_key_is_conversation(self):
        assert ContextSpecification().prompt_key == PromptKey.CONVERSATION

    def test_default_tags_are_empty(self):
        assert ContextSpecification().tags == []

    def test_tags_are_independent_between_instances(self):
        s1 = ContextSpecification()
        s2 = ContextSpecification()
        s1.tags.append("philosophy")
        assert s2.tags == []

    def test_rejects_session_messages_above_max(self):
        with pytest.raises(ValidationError):
            ContextSpecification(last_k_session_messages=21)

    def test_rejects_negative_session_messages(self):
        with pytest.raises(ValidationError):
            ContextSpecification(last_k_session_messages=-1)

    def test_rejects_retrieved_history_above_max(self):
        with pytest.raises(ValidationError):
            ContextSpecification(top_k_retrieved_history=11)


# ═══════════════════════════════════════════════════════════════════════════════
# Fragment & Exchange — UUID generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoGeneratedIds:
    def test_fragment_gets_unique_id_by_default(self):
        f1 = _make_fragment()
        f2 = _make_fragment()
        assert f1.fragment_id != f2.fragment_id

    def test_fragment_id_is_valid_uuid(self):
        f = _make_fragment()
        uuid.UUID(f.fragment_id)  # raises if invalid

    def test_exchange_gets_unique_id_by_default(self):
        e1 = Exchange()
        e2 = Exchange()
        assert e1.exchange_id != e2.exchange_id

    def test_exchange_id_is_valid_uuid(self):
        uuid.UUID(Exchange().exchange_id)

    def test_classified_exchange_gets_unique_id(self):
        now = datetime.now()
        c1 = ClassifiedExchange(
            session_id="s1", exchange_ids=["e1"],
            human_summary="h", ai_summary="a", tags=[], timestamp=now,
        )
        c2 = ClassifiedExchange(
            session_id="s1", exchange_ids=["e1"],
            human_summary="h", ai_summary="a", tags=[], timestamp=now,
        )
        assert c1.classification_id != c2.classification_id


# ═══════════════════════════════════════════════════════════════════════════════
# Turn — auto-timestamp
# ═══════════════════════════════════════════════════════════════════════════════

class TestTurnTimestamp:
    def test_timestamp_defaults_to_now(self):
        before = datetime.now()
        turn = Turn(session_id="s1", role=Role.HUMAN, content="hello")
        after = datetime.now()
        assert before <= turn.timestamp <= after

    def test_explicit_timestamp_is_preserved(self):
        ts = datetime(2024, 1, 1, 12, 0, 0)
        turn = Turn(session_id="s1", role=Role.HUMAN, content="hello", timestamp=ts)
        assert turn.timestamp == ts


# ═══════════════════════════════════════════════════════════════════════════════
# ScoreCard — boundary values
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreCard:
    def test_accepts_boundary_floats(self):
        card = ScoreCard(
            question_score=0.0,
            first_person_score=1.0,
            task_score=0.5,
            domains=[Domain(tag="philosophy", score=0.0), Domain(tag="tech", score=1.0)],
        )
        assert card.question_score == 0.0
        assert card.first_person_score == 1.0

    def test_domains_list_can_be_empty(self):
        card = ScoreCard(question_score=0.5, first_person_score=0.5, task_score=0.5, domains=[])
        assert card.domains == []


# ═══════════════════════════════════════════════════════════════════════════════
# PromptKey — registry completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptKeyRegistry:

    def test_every_prompt_key_resolves_to_a_non_empty_template(self):
        state = _make_state()
        for key in PromptKey:
            template = get_prompt(key, state=state)
            assert isinstance(template, str)
            assert len(template) > 0, f"Empty template for {key}"

    def test_get_prompt_accepts_string_value(self):
        state = _make_state()
        template = get_prompt(PromptKey.CONVERSATION.value, state=state)
        assert isinstance(template, str)

    def test_get_prompt_raises_for_unknown_key(self):
        state = _make_state()
        with pytest.raises(KeyError):
            get_prompt("no_such_key_ever", state=state)

    def test_parametric_prompt_renders_runtime_vars(self):
        profile = UserProfile(human_label="Marker-Alice")
        state = _make_state(user_profile=profile)
        rendered = get_prompt(PromptKey.PROFILE_SCANNER, state=state)
        assert "Marker-Alice" in rendered

    def test_parametric_prompt_missing_var_raises(self):
        with pytest.raises(ValueError):
            get_prompt(PromptKey.PROFILE_SCANNER)


# ═══════════════════════════════════════════════════════════════════════════════
# Status & Role enums — smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnums:
    def test_status_error_value(self):
        assert StatusValue.ERROR == "error"

    def test_role_human_value(self):
        assert Role.HUMAN.value == "human"

    def test_status_is_str_enum(self):
        assert isinstance(StatusValue.COMPLETED, str)
