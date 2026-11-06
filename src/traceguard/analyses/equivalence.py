"""Equivalence testing (TOST) for bounded-null claims.

A central v0.7.0 outcome shape is the BOUNDED NEGATIVE claim: "adding the
reasoning trace does not improve detection beyond actions by more than the
smallest effect size of interest (SESOI)."  A nonsignificant superiority
test cannot license that claim (absence of evidence is not evidence of
absence).  The two-one-sided-tests procedure (Schuirmann 1987; Lakens 2017)
tests H01: delta <= -seoi vs H02: delta >= +seoi and licenses the claim only
when BOTH are rejected at alpha.

Implementation notes:
* the point estimate and CI come from the SAME cluster bootstrap as the
  superiority analysis (frozen thresholds, cluster resampling) — TOST here
  operates on the bootstrap replicate distribution, not on an independent
  asymptotic approximation;
* p-values are computed from the empirical replicate distribution with the
  conservative shift method (equivalent to a percentile-based bound);
* SESOI must be predeclared in the frozen config — never chosen after
  seeing the estimate (recorded as ``seoi_source``).
"""

from __future__ import annotations

from typing import Any


def _fraction_at_or_below(values: list[float], bound: float) -> float:
    if not values:
        return 1.0
    return sum(1 for v in values if v <= bound) / len(values)


def _fraction_at_or_above(values: list[float], bound: float) -> float:
    if not values:
        return 1.0
    return sum(1 for v in values if v >= bound) / len(values)


def tost_from_bootstrap(
    *,
    estimate: float | None,
    bootstrap_deltas: list[float],
    sesoi: float,
    alpha: float = 0.05,
    seoi_source: str = "predeclared_in_frozen_config",
) -> dict[str, Any]:
    """Two-one-sided-tests on the cluster-bootstrap replicate distribution.

    H01: delta <= -sesoi  vs  H02: delta >= +sesoi.  Equivalence is concluded
    only when both one-sided p-values are below alpha AND the point estimate
    is finite.  ``bootstrap_deltas`` are the non-degenerate replicate values
    of the paired difference (frozen-threshold cluster bootstrap).
    """
    if sesoi <= 0:
        raise ValueError("sesoi must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if estimate is None or not bootstrap_deltas:
        return {
            "conclusion": "not_estimable",
            "reason": "missing estimate or bootstrap replicates",
            "sesoi": sesoi,
            "alpha": alpha,
        }
    # Standard TOST with the bootstrap replicate distribution as the
    # sampling distribution of the estimator (centered at the point
    # estimate).  Each one-sided p-value recomputes the tail under the
    # SHIFTED null hypothesis (center -sesoi / +sesoi), which reduces to:
    #   p(H01: delta <= -sesoi rejected when estimate significantly above
    #      -sesoi)  = P(replicate >= 2*estimate + sesoi)
    #   p(H02: delta >= +sesoi rejected when estimate significantly below
    #      +sesoi)  = P(replicate <= 2*estimate - sesoi)
    p_lower = _fraction_at_or_above(bootstrap_deltas, 2.0 * estimate + sesoi)
    p_upper = _fraction_at_or_below(bootstrap_deltas, 2.0 * estimate - sesoi)
    p_tost = max(p_lower, p_upper)
    equivalent = p_lower < alpha and p_upper < alpha
    sorted_deltas = sorted(bootstrap_deltas)
    n = len(sorted_deltas)
    lo = sorted_deltas[int((alpha / 2) * n)]
    hi = sorted_deltas[min(n - 1, int((1 - alpha / 2) * n))]
    return {
        "conclusion": "equivalent_within_sesoi" if equivalent else "not_equivalent",
        "method": "tost_on_cluster_bootstrap_replicates",
        "sesoi": sesoi,
        "seoi_source": seoi_source,
        "alpha": alpha,
        "estimate": estimate,
        "ci_low": lo,
        "ci_high": hi,
        "p_h01_delta_le_minus_sesoi": p_lower,
        "p_h02_delta_ge_plus_sesoi": p_upper,
        "p_tost": p_tost,
        "ci_within_sesoi": bool(lo >= -sesoi and hi <= sesoi),
        "n_bootstrap_replicates": n,
        "interpretation": (
            "the paired marginal value of the reasoning trace is bounded "
            f"within +/- {sesoi} at alpha={alpha} (predeclared SESOI)"
            if equivalent else
            "equivalence within the predeclared SESOI is NOT established; "
            "report the interval, not a claim"
        ),
    }


__all__ = ["tost_from_bootstrap"]
