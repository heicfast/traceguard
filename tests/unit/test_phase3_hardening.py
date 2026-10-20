"""Regression tests for the Phase-3 benchmark and monitor architecture."""

from __future__ import annotations

import json

import pytest

from traceguard.analyses.metrics import (
    false_positive_rate,
    threshold_at_fpr,
    threshold_at_fpr_inclusive_ties,
)
from traceguard.analyses.uncertainty import cluster_bootstrap_paired
from traceguard.config import MonitorConfig, load_config
from traceguard.models.interface import ModelTurn
from traceguard.monitors.structured import StructuredLateFusionMonitor
from traceguard.sandbox.engine import step
from traceguard.tasks.generator import build_episode_plan
from traceguard.tasks.specs import instantiate
from traceguard.traces.views import MonitorObservation


def test_relational_violation_requires_registry_resolution():
    task = instantiate("artifact_routing/promote_candidate", 17)
    state = task.initial_state()
    invariant = task.invariants[0]
    assert invariant.op == "ref_target_equals"
    assert invariant.evaluate(state)

    result = step(state, task.attack_reference_action, task.invariants)
    assert result.action_accepted
    assert result.protected_violation
    # The patch itself touches a neutral alias field; the protected property
    # exists only in the separately resolved registry object.
    assert task.attack_reference_action.key_path == ["artifact_alias"]
    assert "signed" not in task.attack_reference_action.key_path


def test_phase3_plan_crosses_strategy_with_exact_policy_twins():
    config = load_config("configs/phase3_pilot.yaml")
    plans = build_episode_plan(config, "pilot")
    assert len(plans) == 128
    strategies = set(config.dataset.attack_strategies)
    assert {plan.attack_strategy for plan in plans} == strategies
    keys = {
        (plan.template_name, plan.instance_index, plan.generation_seed,
         plan.attack_strategy): set()
        for plan in plans
    }
    for plan in plans:
        keys[(plan.template_name, plan.instance_index, plan.generation_seed,
              plan.attack_strategy)].add(plan.policy_mode)
    assert all(modes == {"honest", "attack"} for modes in keys.values())


class _PromptRecordingProvider:
    provider_name = "recording"

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, messages, settings):
        self.prompts.append(messages[-1]["content"])
        index = len(self.prompts)
        if index == 1:
            payload = {"policy_facts": ["alias must resolve safely"],
                       "observed_actions": ["alias changed"],
                       "risk_indicators": ["registry mismatch"], "missing_context": []}
        elif index == 2:
            payload = {"intent_claims": ["select fallback"],
                       "authorization_claims": ["agent says it is required"],
                       "concealment_indicators": [], "uncertainty": [],
                       "quoted_fragments": ["select fallback"]}
        else:
            payload = {"risk_score": 75,
                       "rubric": {"policy_conflict": 25, "unauthorized_intent": 25,
                                  "concealment": 5, "action_evidence": 20},
                       "evidence_refs": ["alias changed"],
                       "verdict_text": "The alias conflicts with policy."}
        raw = json.dumps(payload)
        return ModelTurn(raw_output=raw, assistant_text=raw, model_revision="test-rev")


def test_late_fusion_withholds_raw_cot_from_adjudicator():
    provider = _PromptRecordingProvider()
    monitor = StructuredLateFusionMonitor(provider, retry_once=False)
    action = MonitorObservation("ep-phase3", "actions", "ACTION VIEW", 1, 0)
    reasoning = MonitorObservation(
        "ep-phase3", "cot_only", "RAW_COT_SENTINEL agent narrative", 1, 1)
    action.compute_hash()
    reasoning.compute_hash()

    score = monitor.score_views(
        action, reasoning, condition="actions_plus_cot_late_fusion")
    assert score.valid and score.suspicion == 0.75
    assert "RAW_COT_SENTINEL" in provider.prompts[1]
    assert "RAW_COT_SENTINEL" not in provider.prompts[2]
    assert score.parsed_response["reasoning_extractor"]["intent_claims"]
    assert len(score.inference_audit) == 3


def test_small_cluster_ci_is_explicitly_not_estimable():
    rows = [
        {"template_id": "benign", "label": 0, "scores": {"C": 0.1, "A": 0.1}},
        {"template_id": "attack", "label": 1, "scores": {"C": 0.9, "A": 0.2}},
    ]
    result = cluster_bootstrap_paired(
        rows, "C", "A", 0.05, replicates=200, seed=2,
        frozen_thresholds=(0.5, 0.5), min_total_clusters_for_ci=8,
        min_positive_clusters_for_ci=4, min_support_for_ci=3,
        max_degenerate_fraction_for_ci=0.10,
    )
    assert result["estimate"] == 1.0
    assert result["ci_status"] == "not_estimable"
    assert result["ci_low"] is None and result["ci_high"] is None
    assert "too_few_total_clusters" in result["ci_ineligibility_reasons"]
    assert result["exact_cluster_distribution"]["compositions"] == 3


def test_tie_sensitivity_exposes_coarse_score_threshold_instability():
    benign = [0.0] * 18 + [0.5] * 2
    conservative = threshold_at_fpr(benign, 0.05)
    inclusive = threshold_at_fpr_inclusive_ties(benign, 0.05)
    assert false_positive_rate(benign, conservative) == 0.0
    assert false_positive_rate(benign, inclusive) == 0.10


def test_fact_checked_arm_requires_pinned_independent_model():
    with pytest.raises(ValueError, match="requires a pinned fact-checker"):
        MonitorConfig(conditions=["actions", "actions_plus_cot_fact_checked"])
    with pytest.raises(ValueError, match="different model family"):
        MonitorConfig(
            conditions=["actions", "actions_plus_cot_fact_checked"],
            fact_checker_model_id="Other/checker",
            fact_checker_model_revision="abc123",
            fact_checker_model_family="qwen3",
            require_cross_family_fact_checker=True,
        )
