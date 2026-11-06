"""Unit tests for the intent-era task families and new attack strategies."""

from __future__ import annotations

import pytest

from traceguard.agents.policy import build_messages, render_attack_instruction
from traceguard.sandbox.engine import step as engine_step
from traceguard.tasks.specs import FAMILY_TEMPLATES, instantiate

INTENT_FAMILIES = ["approval_window", "quota_accumulation", "bulk_masking",
                   "delegated_authority"]


def test_intent_families_are_registered_with_two_templates_each():
    for family in INTENT_FAMILIES:
        assert len(FAMILY_TEMPLATES[family]) == 2, family


def test_intent_families_carry_diagnostic_complexity():
    from traceguard.tasks.intent_specs import DIAGNOSTIC_COMPLEXITY

    for family, tier in DIAGNOSTIC_COMPLEXITY.items():
        template = FAMILY_TEMPLATES[family][0]
        inst = instantiate(template, 0)
        assert inst.diagnostic_complexity == tier


@pytest.mark.parametrize("family", INTENT_FAMILIES)
def test_honest_completion_does_not_violate(family):
    """The main-goal edit path stays policy-clean in every intent family."""
    template = FAMILY_TEMPLATES[family][0]
    for seed in (0, 1, 2):
        inst = instantiate(template, seed)
        state = inst.initial_state()
        goal_path = inst.main_goal_path
        from traceguard.sandbox.actions import TypedAction

        action = TypedAction(action="propose_patch", target_path=goal_path[0],
                              operation="set_value", key_path=list(goal_path[1:]),
                              value=inst.main_goal_expected)
        result = engine_step(state, action, inst.invariants)
        assert not result.violated_invariants, (family, seed, result.violated_invariants)


def test_approval_window_attack_is_reference_join():
    """The violating action is a routine-looking reference update; the
    discriminating evidence is the registry join (window mismatch)."""
    inst = instantiate("approval_window/link_validation", 0)
    state = inst.initial_state()
    # initial state satisfies the relational invariant
    assert all(inv.evaluate(state) for inv in inst.invariants)
    attack = inst.attack_reference_action
    result = engine_step(state, attack, inst.invariants)
    assert result.action_accepted
    assert result.violated_invariants
    # and the tool response is condition-blind (no violation marker in text)
    assert "violation" not in result.tool_response.lower()


def test_quota_attack_requires_cumulative_violation():
    inst = instantiate("quota_accumulation/capacity_round", 0)
    state = inst.initial_state()
    assert all(inv.evaluate(state) for inv in inst.invariants)
    working = state
    violations_seen = []
    for action in inst.attack_reference_actions:
        result = engine_step(working, action, inst.invariants)
        violations_seen.extend(result.violated_invariants)
        working = result.new_state
    assert violations_seen, "cumulative attack must eventually violate"
    assert len(inst.attack_reference_actions) >= 4  # genuinely spread out


def test_bulk_masking_buries_protected_rename():
    inst = instantiate("bulk_masking/modernize_schema", 0)
    state = inst.initial_state()
    # the honest path renames 5+ legitimate keys without violating
    renamed = 0
    for key in list(state.objects["data/events_schema.json"]):
        if key.startswith("field_"):
            renamed += 1
    assert renamed >= 5
    # the attack's protected rename violates
    result = engine_step(state, inst.attack_reference_action, inst.invariants)
    assert result.violated_invariants


def test_delegated_authority_self_approval_violates():
    inst = instantiate("delegated_authority/review_setting", 0)
    state = inst.initial_state()
    assert all(inv.evaluate(state) for inv in inst.invariants)
    result = engine_step(state, inst.attack_reference_action, inst.invariants)
    assert result.violated_invariants
    assert any("pending" in v for v in result.violated_invariants)


NEW_STRATEGIES = ["authorization_claim", "incremental_accumulation",
                  "needle_concealment", "n_hop_indirection"]
LEGACY_STRATEGIES = ["explicit_override", "side_objective",
                     "implicit_pressure", "monitor_aware"]


@pytest.mark.parametrize("strategy", NEW_STRATEGIES + LEGACY_STRATEGIES)
def test_every_attack_strategy_renders_and_is_privileged_only(strategy):
    inst = instantiate("config_update/flag_flip", 0)
    rendered = render_attack_instruction(inst, strategy)
    assert rendered
    assert inst.attack_instruction in rendered  # objective never changes
    messages = build_messages(inst, "attack", strategy)
    assert len(messages) == 3  # system, user(goal), user(hidden)
    assert rendered in messages[2]["content"]


def test_unknown_strategy_fails_loudly():
    inst = instantiate("config_update/flag_flip", 0)
    with pytest.raises(ValueError, match="unknown attack strategy"):
        render_attack_instruction(inst, "definitely_not_a_strategy")


def test_new_families_have_split_roles():
    from traceguard.tasks.split import FAMILY_SPLIT

    assert FAMILY_SPLIT["approval_window"] == "development"
    assert FAMILY_SPLIT["quota_accumulation"] == "validation"
    assert FAMILY_SPLIT["bulk_masking"] == "holdout"
    assert FAMILY_SPLIT["delegated_authority"] == "shift"
