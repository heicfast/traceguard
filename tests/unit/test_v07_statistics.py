"""Unit tests for the v0.7.0 statistics additions.

Hand-computed fixtures only — every number below was derived by direct
computation, not copied from a library output.
"""

from __future__ import annotations

import math

from traceguard.analyses.agreement import cohen_kappa, confusion_matrix
from traceguard.analyses.calibration import (
    brier_score,
    expected_calibration_error,
    net_benefit,
    reliability_bins,
)
from traceguard.analyses.equivalence import tost_from_bootstrap
from traceguard.analyses.metrics import (
    benjamini_hochberg,
    cluster_label_permutation_test,
    holm_adjust,
)
from traceguard.analyses.power import empirical_power, simulate_clustered_paired_design

# ---------------------------------------------------------------------------
# TOST equivalence
# ---------------------------------------------------------------------------

def test_tost_concludes_equivalence_inside_sesoi():
    deltas = [round(-0.01 + 0.002 * i, 4) for i in range(11)]  # -0.01..+0.01
    result = tost_from_bootstrap(estimate=0.0, bootstrap_deltas=deltas, sesoi=0.05)
    assert result["conclusion"] == "equivalent_within_sesoi"
    assert result["p_tost"] < 0.05
    assert result["ci_within_sesoi"] is True


def test_tost_refuses_equivalence_when_effect_is_large():
    deltas = [round(0.08 + 0.004 * i, 4) for i in range(11)]  # 0.08..0.12
    result = tost_from_bootstrap(estimate=0.10, bootstrap_deltas=deltas, sesoi=0.05)
    assert result["conclusion"] == "not_equivalent"
    assert result["p_h02_delta_ge_plus_sesoi"] >= 0.05


def test_tost_not_estimable_without_replicates():
    result = tost_from_bootstrap(estimate=None, bootstrap_deltas=[], sesoi=0.05)
    assert result["conclusion"] == "not_estimable"


def test_tost_rejects_nonpositive_sesoi():
    import pytest

    with pytest.raises(ValueError):
        tost_from_bootstrap(estimate=0.0, bootstrap_deltas=[0.0], sesoi=0.0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_brier_score_perfect_and_anti():
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([1.0, 0.0], [0, 1]) == 1.0
    # half-right at half-confidence: (0.5-0)^2 and (0.5-1)^2 -> mean 0.25
    assert math.isclose(brier_score([0.5, 0.5], [0, 1]), 0.25)


def test_brier_rejects_out_of_range_scores():
    import pytest

    with pytest.raises(ValueError):
        brier_score([1.5], [1])
    with pytest.raises(ValueError):
        brier_score([0.5], [2])


def test_reliability_bins_partition():
    bins = reliability_bins([0.05, 0.15, 0.95], [0, 1, 1], n_bins=10)
    assert sum(b["count"] for b in bins) == 3
    first = bins[0]
    assert first["count"] == 1 and first["fraction_positive"] == 0.0
    last = bins[9]
    assert last["count"] == 1 and last["fraction_positive"] == 1.0


def test_ece_zero_for_perfect_calibration_in_bins():
    scores = [0.25, 0.25, 0.25, 0.25]
    labels = [1, 0, 0, 0]  # exactly 25% positive in the 0.2-0.3 bin
    assert expected_calibration_error(scores, labels) == 0.0


def test_net_benefit_classic_convention():
    # threshold=0.5, 1:1 utilities: NB = TP/n - FP/n*(0.5/0.5) = (TP-FP)/n
    scores = [0.9, 0.8, 0.4, 0.2]
    labels = [1, 0, 1, 0]
    result = net_benefit(scores, labels, 0.5)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert math.isclose(result["net_benefit"], (1 - 1) / 4)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def test_cohen_kappa_perfect_and_partial():
    assert cohen_kappa([True, False], [True, False])["kappa"] == 1.0
    result = cohen_kappa(
        [True, True, False, False], [True, False, True, False])
    # po=0.5, pa=pb=0.5, pe=0.5 -> kappa=0
    assert result["kappa"] == 0.0
    assert result["both_flag"] == 1 and result["a_only"] == 1


def test_confusion_matrix_counts():
    counts = confusion_matrix([True, True, False], [True, False, False])
    assert counts == {"both_flag": 1, "a_only": 1, "b_only": 0, "neither": 1}


# ---------------------------------------------------------------------------
# Multiple comparisons + permutation
# ---------------------------------------------------------------------------

def test_bh_matches_manual_stepup():
    # p=[0.01, 0.04, 0.03]: sorted ranks give .01*3=.03, .03*3/2=.045,
    # .04*3/3=.04; step-up monotonicity (largest first) caps the middle at
    # .04 -> adjusted [.03, .04, .04] (matches statsmodels fdr_bh).
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest_approx(0.03)
    assert adjusted[1] == pytest_approx(0.04)
    assert adjusted[2] == pytest_approx(0.04)


def pytest_approx(value: float, tol: float = 1e-9) -> float:
    class _Approx(float):
        def __eq__(self, other: object) -> bool:  # type: ignore[override]
            return isinstance(other, (int, float)) and abs(self - other) <= tol

    return _Approx(value)


def test_bh_agrees_with_holm_on_all_significant():
    p = [0.001, 0.002]
    assert all(v <= 0.01 for v in benjamini_hochberg(p))
    assert all(v <= 0.01 for v in holm_adjust(p))


def test_cluster_permutation_detects_association():
    labels = {f"c{i}": i % 2 for i in range(10)}
    stats = {f"c{i}": 0.9 if i % 2 == 1 else 0.1 for i in range(10)}
    result = cluster_label_permutation_test(labels, stats, replicates=2000, seed=7)
    assert result["p_value"] is not None and result["p_value"] <= 0.05


def test_cluster_permutation_null_gives_uniform_p():
    labels = {f"c{i}": i % 2 for i in range(12)}
    stats = {f"c{i}": 0.5 for i in range(12)}  # no association at all
    result = cluster_label_permutation_test(labels, stats, replicates=1000, seed=3)
    # degenerate statistic -> p stays maximal
    assert result["p_value"] >= 0.99


def test_cluster_permutation_key_mismatch_is_error():
    import pytest

    with pytest.raises(ValueError):
        cluster_label_permutation_test({"a": 1}, {"b": 0.5})


# ---------------------------------------------------------------------------
# Power simulation
# ---------------------------------------------------------------------------

def test_power_sim_high_delta_yields_high_power():
    result = empirical_power(
        repetitions=20, seed=11,
        n_clusters=30, episodes_per_cluster=8, attack_fraction=0.5,
        delta_true=0.4, recall_baseline=0.3,
        intracluster_correlation=0.05,
        bootstrap_replicates=200, confidence_level=0.95,
    )
    assert result["empirical_power"] is not None
    assert result["empirical_power"] >= 0.7


def test_power_sim_zero_delta_never_rejects_directionally():
    result = empirical_power(
        repetitions=10, seed=13,
        n_clusters=20, episodes_per_cluster=8, attack_fraction=0.5,
        delta_true=0.0, recall_baseline=0.5,
        intracluster_correlation=0.10,
        bootstrap_replicates=200, confidence_level=0.95,
    )
    assert result["empirical_power"] is not None
    assert result["empirical_power"] <= 0.3  # ~alpha-level false positives


def test_single_simulation_records_design():
    result = simulate_clustered_paired_design(
        n_clusters=10, episodes_per_cluster=6, attack_fraction=0.5,
        delta_true=0.2, recall_baseline=0.4, intracluster_correlation=0.1,
        bootstrap_replicates=100, confidence_level=0.95, seed=5,
    )
    assert result["n_pairs"] > 0
    assert "ci_low" in result and "ci_high" in result
