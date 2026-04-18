"""Layer 4 tests — ScoreCard resolution (configure/score_card.py)."""

import itertools

import pytest

from journal_agent.configure.score_card import (
    INTENT_TO_SPEC,
    THRESHOLDS,
    Intent,
    resolve_scorecard_to_specification,
    _DEFAULT_SPEC,
    _GUIDANCE_SPEC,
    _SOCRATIC_SPEC,
)
from journal_agent.model.session import ContextSpecification, Domain, PromptKey, ScoreCard


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_card(
    q: float = 0.0,
    fp: float = 0.0,
    t: float = 0.0,
    domains: list[Domain] | None = None,
) -> ScoreCard:
    return ScoreCard(
        question_score=q,
        first_person_score=fp,
        task_score=t,
        domains=domains or [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Intent enum — structural completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntentEnum:
    def test_all_8_boolean_triples_map_to_a_valid_intent(self):
        for combo in itertools.product([True, False], repeat=3):
            intent = Intent(combo)
            assert isinstance(intent, Intent)

    def test_intent_enum_has_exactly_8_members(self):
        assert len(Intent) == 8

    def test_intent_to_spec_covers_six_intents(self):
        assert len(INTENT_TO_SPEC) == 6

    def test_seeking_help_and_curious_are_not_in_explicit_map(self):
        assert Intent.SEEKING_HELP not in INTENT_TO_SPEC
        assert Intent.CURIOUS not in INTENT_TO_SPEC

    def test_every_mapped_intent_has_a_context_specification(self):
        for intent, spec in INTENT_TO_SPEC.items():
            assert isinstance(spec, ContextSpecification), f"{intent} has invalid spec"


# ═══════════════════════════════════════════════════════════════════════════════
# resolve_scorecard_to_specification — all 8 intent paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveAllIntents:
    def test_observing_resolves_to_socratic(self):
        # (False, False, False) → OBSERVING → SOCRATIC
        spec = resolve_scorecard_to_specification(make_card(0.0, 0.0, 0.0))
        assert spec.prompt_key == PromptKey.SOCRATIC

    def test_musing_resolves_to_socratic(self):
        # (False, True, False) → MUSING → SOCRATIC
        spec = resolve_scorecard_to_specification(make_card(0.0, 1.0, 0.0))
        assert spec.prompt_key == PromptKey.SOCRATIC

    def test_self_questioning_resolves_to_socratic(self):
        # (True, True, False) → SELF_QUESTIONING → SOCRATIC
        spec = resolve_scorecard_to_specification(make_card(1.0, 1.0, 0.0))
        assert spec.prompt_key == PromptKey.SOCRATIC

    def test_directing_resolves_to_guidance(self):
        # (False, False, True) → DIRECTING → GUIDANCE
        spec = resolve_scorecard_to_specification(make_card(0.0, 0.0, 1.0))
        assert spec.prompt_key == PromptKey.GUIDANCE

    def test_planning_resolves_to_guidance(self):
        # (False, True, True) → PLANNING → GUIDANCE
        spec = resolve_scorecard_to_specification(make_card(0.0, 1.0, 1.0))
        assert spec.prompt_key == PromptKey.GUIDANCE

    def test_researching_resolves_to_guidance(self):
        # (True, False, True) → RESEARCHING → GUIDANCE
        spec = resolve_scorecard_to_specification(make_card(1.0, 0.0, 1.0))
        assert spec.prompt_key == PromptKey.GUIDANCE

    def test_curious_resolves_to_default(self):
        # (True, False, False) → CURIOUS → DEFAULT fallback
        spec = resolve_scorecard_to_specification(make_card(1.0, 0.0, 0.0))
        assert spec.prompt_key == PromptKey.CONVERSATION

    def test_seeking_help_resolves_to_default(self):
        # (True, True, True) → SEEKING_HELP → DEFAULT fallback
        spec = resolve_scorecard_to_specification(make_card(1.0, 1.0, 1.0))
        assert spec.prompt_key == PromptKey.CONVERSATION

    def test_returns_context_specification_for_all_8_intents(self):
        for combo in itertools.product([0.0, 1.0], repeat=3):
            card = make_card(*combo)
            spec = resolve_scorecard_to_specification(card)
            assert isinstance(spec, ContextSpecification)


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold boundary conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdBoundary:
    def test_score_exactly_at_threshold_is_not_above(self):
        # 0.5 is NOT > 0.5 — stays False, same result as 0.0
        spec_at = resolve_scorecard_to_specification(make_card(q=0.5, fp=0.0, t=0.0))
        spec_below = resolve_scorecard_to_specification(make_card(q=0.0, fp=0.0, t=0.0))
        assert spec_at.prompt_key == spec_below.prompt_key

    def test_score_just_above_threshold_flips_to_true(self):
        # 0.51 > 0.5 — flips question to True → different intent path
        spec_just_over = resolve_scorecard_to_specification(make_card(q=0.51, fp=0.0, t=0.0))
        spec_below = resolve_scorecard_to_specification(make_card(q=0.0, fp=0.0, t=0.0))
        # (True, False, False) = CURIOUS → DEFAULT; (False, False, False) = OBSERVING → SOCRATIC
        assert spec_just_over.prompt_key != spec_below.prompt_key

    def test_thresholds_dict_values_are_all_0_5(self):
        for key, value in THRESHOLDS.items():
            assert value == 0.5, f"Unexpected threshold for {key}: {value}"


# ═══════════════════════════════════════════════════════════════════════════════
# Domain tag propagation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainTagPropagation:
    def test_high_scoring_domains_appear_in_spec_tags(self):
        domains = [Domain(tag="philosophy", score=1.0), Domain(tag="tech", score=0.8)]
        spec = resolve_scorecard_to_specification(make_card(domains=domains))
        assert "philosophy" in spec.tags
        assert "tech" in spec.tags

    def test_low_scoring_domains_are_excluded(self):
        domains = [Domain(tag="excluded", score=0.3)]
        spec = resolve_scorecard_to_specification(make_card(domains=domains))
        assert "excluded" not in spec.tags

    def test_domain_score_exactly_at_0_5_is_excluded(self):
        domains = [Domain(tag="borderline", score=0.5)]
        spec = resolve_scorecard_to_specification(make_card(domains=domains))
        assert "borderline" not in spec.tags

    def test_empty_domains_produces_empty_tags(self):
        spec = resolve_scorecard_to_specification(make_card())
        assert spec.tags == []


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton mutation — returned spec must be a copy, not the shared instance
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingletonIsolation:
    def test_returned_spec_is_not_the_shared_socratic_singleton(self):
        spec = resolve_scorecard_to_specification(make_card(0.0, 0.0, 0.0))  # OBSERVING → SOCRATIC
        assert spec is not _SOCRATIC_SPEC

    def test_returned_spec_is_not_the_shared_guidance_singleton(self):
        spec = resolve_scorecard_to_specification(make_card(0.0, 0.0, 1.0))  # DIRECTING → GUIDANCE
        assert spec is not _GUIDANCE_SPEC

    def test_returned_spec_is_not_the_shared_default_singleton(self):
        spec = resolve_scorecard_to_specification(make_card(1.0, 0.0, 0.0))  # CURIOUS → DEFAULT
        assert spec is not _DEFAULT_SPEC

    def test_setting_tags_does_not_mutate_socratic_singleton(self):
        original_tags = list(_SOCRATIC_SPEC.tags)
        domains = [Domain(tag="philosophy", score=1.0)]
        resolve_scorecard_to_specification(make_card(0.0, 0.0, 0.0, domains=domains))
        assert _SOCRATIC_SPEC.tags == original_tags

    def test_setting_tags_does_not_mutate_guidance_singleton(self):
        original_tags = list(_GUIDANCE_SPEC.tags)
        domains = [Domain(tag="philosophy", score=1.0)]
        resolve_scorecard_to_specification(make_card(0.0, 0.0, 1.0, domains=domains))
        assert _GUIDANCE_SPEC.tags == original_tags

    def test_two_calls_with_different_domains_return_independent_specs(self):
        d1 = [Domain(tag="philosophy", score=1.0)]
        d2 = [Domain(tag="tech", score=1.0)]
        spec1 = resolve_scorecard_to_specification(make_card(0.0, 0.0, 0.0, domains=d1))
        spec2 = resolve_scorecard_to_specification(make_card(0.0, 0.0, 0.0, domains=d2))
        assert spec1.tags == ["philosophy"]
        assert spec2.tags == ["tech"]
