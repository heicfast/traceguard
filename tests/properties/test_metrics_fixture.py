"""Property + fixture tests: statistical metrics vs hand-computed values,
cluster bootstrap behavior, task-level attack simulation across the registry."""

from __future__ import annotations

import math

import pytest

from traceguard.analyses.metrics import (
    auroc,
    average_precision,
    false_positive_rate,
    recall_at_threshold,
    roc_points,
    threshold_at_fpr,
)
from traceguard.analyses.uncertainty import cluster_bootstrap_statistic
from traceguard.sandbox.engine import simulate_proposed_patch_violation
from traceguard.tasks.specs import TEMPLATES, instantiate

# -- hand-computed metric fixtures --------------------------------------------

BENIGN = [0.10, 0.20, 0.30, 0.40, 0.50]
ATTACK = [0.35, 0.55, 0.65, 0.75, 0.95]


def test_threshold_at_fpr_zero():
    t = threshold_at_fpr(BENIGN, 0.0)
    assert false_positive_rate(BENIGN, t) == 0.0
    assert t > 0.50


def test_threshold_at_fpr_20pct():
    # allow 1 FP of 5 benign -> threshold above the 4th largest (0.40)
    t = threshold_at_fpr(BENIGN, 0.2)
    fps = sum(1 for s in BENIGN if s >= t)
    assert fps <= 1
    assert t > 0.40 and t <= 0.50 + 1e-9


def test_threshold_at_fpr_full():
    t = threshold_at_fpr(BENIGN, 1.0)
    assert math.isinf(t) or t <= -1e18
    assert recall_at_threshold(ATTACK, t) == 1.0


def test_recall_matches_hand_computation():
    t = threshold_at_fpr(BENIGN, 0.2)  # t slightly > 0.40
    # attacks >= t: 0.50? no (0.50 < t just above 0.40... wait t>0.40, 0.50>=t yes)
    recalled = [s for s in ATTACK if s >= t]
    assert len(recalled) == 4  # 0.50, 0.65, 0.75, 0.95 ... check 0.55 too
    # precise: t just above 0.40 => 0.50,0.55,0.65,0.75,0.95 all >= t => 5
    assert recall_at_threshold(ATTACK, t) == len(recalled) / len(ATTACK)


def test_auroc_perfect_and_random():
    assert auroc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert auroc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auroc_hand_value():
    # positives: 0.6, 0.4; negatives: 0.5, 0.3
    # ranks: 0.3->1, 0.4->2, 0.5->3, 0.6->4; U = (2+4) - 3 = 3; AUC = 3/4
    assert auroc([0.6, 0.4], [0.5, 0.3]) == pytest.approx(0.75)


def test_auroc_with_ties():
    # ties split credit: positives [0.5], negatives [0.5] -> AUC = 0.5
    assert auroc([0.5], [0.5]) == 0.5
    # positives [1.0, 0.5], negatives [0.5, 0.0]:
    # ranks: 0.0->1, 0.5->2.5, 0.5->2.5, 1.0->4; pos ranks 2.5 + 4 = 6.5
    # U = 6.5 - 3 = 3.5; AUC = 3.5/4 = 0.875
    assert auroc([1.0, 0.5], [0.5, 0.0]) == pytest.approx(0.875)


def test_average_precision_extremes():
    assert average_precision([0.9], [0.1]) == 1.0
    # a single positive ranked last among 2: AP = 1/2 (precision at the hit)
    assert average_precision([0.1], [0.9]) == pytest.approx(0.5)


def test_auroc_and_ap_match_sklearn():
    """Cross-check our pure implementations against sklearn on random cases."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        pytest.skip("sklearn unavailable")
    import random
    rng = random.Random(7)
    for _ in range(20):
        n_pos = rng.randint(1, 8)
        n_neg = rng.randint(1, 8)
        pos = [round(rng.random(), 3) for _ in range(n_pos)]
        neg = [round(rng.random(), 3) for _ in range(n_neg)]
        labels = [1] * n_pos + [0] * n_neg
        scores = pos + neg
        assert auroc(pos, neg) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        assert average_precision(pos, neg) == pytest.approx(
            average_precision_score(labels, scores), abs=1e-12)


def test_roc_points_monotone():
    pts = roc_points(ATTACK, BENIGN)
    for (f1, t1), (f2, t2) in zip(pts, pts[1:], strict=False):
        assert f2 >= f1 and t2 >= t1


def test_threshold_validation():
    with pytest.raises(ValueError):
        threshold_at_fpr([], 0.1)
    with pytest.raises(ValueError):
        threshold_at_fpr(BENIGN, 1.5)


# -- cluster bootstrap ---------------------------------------------------------

def _rows(template, scores, labels):
    return [{"template_id": template, "family_id": "f", "label": lab,
             "scores": scores} for lab in labels]


def test_cluster_bootstrap_degenerate_cluster():
    """If one of two clusters has no benign rows, replicates without any benign
    rows are dropped explicitly (degenerate count reported)."""
    rows = [
        {"template_id": "t1", "family_id": "f1", "label": 0, "scores": {"C": 0.1, "A": 0.1}},
        {"template_id": "t1", "family_id": "f1", "label": 1, "scores": {"C": 0.9, "A": 0.9}},
        {"template_id": "t2", "family_id": "f2", "label": 1, "scores": {"C": 0.8, "A": 0.8}},
    ]
    def stat(sample):
        has_pos = any(r["label"] == 1 for r in sample)
        has_neg = any(r["label"] == 0 for r in sample)
        if not (has_pos and has_neg):
            return None
        return auroc([r["scores"]["C"] for r in sample if r["label"] == 1],
                     [r["scores"]["C"] for r in sample if r["label"] == 0])
    result = cluster_bootstrap_statistic(rows, stat, cluster_key="template_id",
                                          replicates=50, seed=1)
    assert result["estimate"] == 1.0
    assert result["ci_low"] is not None
    assert result["degenerate_replicates"] > 0


def test_cluster_bootstrap_deterministic():
    rows = [
        {"template_id": f"t{i}", "family_id": "f", "label": i % 2,
         "scores": {"C": 0.2 + 0.05 * i, "A": 0.2 + 0.05 * i}}
        for i in range(10)
    ]
    def stat(sample):
        return sum(r["scores"]["C"] for r in sample) / len(sample)

    r1 = cluster_bootstrap_statistic(rows, stat, replicates=30, seed=99)
    r2 = cluster_bootstrap_statistic(rows, stat, replicates=30, seed=99)
    assert r1["ci_low"] == r2["ci_low"] and r1["ci_high"] == r2["ci_high"]


# -- whole-registry attack simulation -------------------------------------------

@pytest.mark.parametrize("template_name", sorted(TEMPLATES))
def test_reference_attack_violates_for_all_templates(template_name):
    for seed in (0, 1, 2, 3):
        inst = instantiate(template_name, seed)
        state = inst.initial_state()
        violations = simulate_proposed_patch_violation(
            state, inst.attack_reference_action, inst.invariants)
        assert violations, f"{template_name} seed={seed}: reference attack has no effect"


@pytest.mark.parametrize("template_name", sorted(TEMPLATES))
def test_main_goal_edit_is_benign_for_all_templates(template_name):
    """The legitimate main-goal edit must NOT violate any invariant."""
    for seed in (0, 1):
        inst = instantiate(template_name, seed)
        state = inst.initial_state()
        from traceguard.sandbox.actions import TypedAction
        goal_path = inst.main_goal_path
        legit = TypedAction(action="propose_patch", target_path=goal_path[0],
                             operation="set_value", key_path=list(goal_path[1:]),
                             value=inst.main_goal_expected)
        violations = simulate_proposed_patch_violation(state, legit, inst.invariants)
        assert not violations, (
            f"{template_name} seed={seed}: legitimate goal edit violates protection: {violations}")
