"""Uncertainty quantification: cluster bootstrap for paired comparisons.

Design (docs/01 §10):
* Resampling unit = the CLUSTER (template by default, family for shift
  analyses), never the individual episode, because episodes of one template
  are near-duplicates and five monitors scoring one episode are not five
  independent observations.
* Paired conditions are resampled jointly (same clusters for both arms), so
  the within-cluster correlation between conditions is preserved.
* Fixed seed, replicates count, and percentile method are frozen in the
  config and recorded in every result payload.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from itertools import combinations_with_replacement
from math import factorial
from typing import Any

from .metrics import threshold_at_fpr


def cluster_bootstrap_paired(
    rows: list[dict[str, Any]],
    condition_primary: str,
    condition_baseline: str,
    fpr_target: float,
    *,
    cluster_key: str = "template_id",
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20261019,
    split_filter: str | None = None,
    frozen_thresholds: tuple[float, float] | None = None,
    min_total_clusters_for_ci: int | None = None,
    min_positive_clusters_for_ci: int | None = None,
    max_degenerate_fraction_for_ci: float | None = None,
    min_support_for_ci: int | None = None,
) -> dict[str, Any]:
    """Bootstrap the paired recall difference at a calibrated FPR.

    ``frozen_thresholds`` (primary, baseline) pins the decision thresholds to
    the pre-registered calibration-split values so that every replicate — and
    the CI as a whole — characterizes THE SAME estimand as the point estimate
    (review finding 2026-10-19: recalibrating thresholds inside each replicate
    on the evaluation split's own benign rows characterized a different,
    within-replicate-recalibrated quantity). When ``frozen_thresholds`` is
    None, the legacy within-replicate recalibration is retained for backwards
    compatibility and the result is labeled accordingly. Replicates without
    attack rows contribute no recall statistic and are counted as degenerate.
    """
    if split_filter:
        rows = [r for r in rows if r["split_role"] == split_filter]
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(row[cluster_key], []).append(row)
    keys = sorted(clusters)
    if not keys:
        return {"estimate": None, "ci_low": None, "ci_high": None, "error": "no_data"}

    def compute_delta(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
        benign = [r for r in sample_rows if r["label"] == 0]
        attacks = [r for r in sample_rows if r["label"] == 1]
        if not attacks:
            return {"delta": None, "recall_primary": None, "recall_baseline": None}
        if frozen_thresholds is not None:
            t_p, t_b = frozen_thresholds
        else:
            if not benign:
                return {"delta": None, "recall_primary": None, "recall_baseline": None}
            t_p = threshold_at_fpr([r["scores"][condition_primary] for r in benign], fpr_target)
            t_b = threshold_at_fpr([r["scores"][condition_baseline] for r in benign], fpr_target)
        rec_p = sum(1 for r in attacks if r["scores"][condition_primary] >= t_p) / len(attacks)
        rec_b = sum(1 for r in attacks if r["scores"][condition_baseline] >= t_b) / len(attacks)
        return {"delta": rec_p - rec_b, "recall_primary": rec_p, "recall_baseline": rec_b}

    point = compute_delta(rows)
    rng = random.Random(seed)
    deltas: list[float] = []
    nan_replicates = 0
    for _ in range(replicates):
        sample: list[dict[str, Any]] = []
        for _ in range(len(keys)):
            sample.extend(clusters[rng.choice(keys)])
        stat = compute_delta(sample)
        if stat["delta"] is None:
            nan_replicates += 1
            continue
        deltas.append(stat["delta"])
    if not deltas:
        return {"estimate": None, "ci_low": None, "ci_high": None,
                "error": "all_replicates_degenerate", "point": point}
    deltas.sort()
    alpha = 1.0 - confidence_level
    lo = deltas[int(alpha / 2 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha / 2) * len(deltas)))]
    positive_clusters = len({r[cluster_key] for r in rows if r["label"] == 1})
    support = len(set(deltas))
    degenerate_fraction = nan_replicates / replicates
    ci_reasons: list[str] = []
    if min_total_clusters_for_ci is not None and len(keys) < min_total_clusters_for_ci:
        ci_reasons.append("too_few_total_clusters")
    if (min_positive_clusters_for_ci is not None
            and positive_clusters < min_positive_clusters_for_ci):
        ci_reasons.append("too_few_positive_clusters")
    if (max_degenerate_fraction_for_ci is not None
            and degenerate_fraction > max_degenerate_fraction_for_ci):
        ci_reasons.append("too_many_degenerate_replicates")
    if min_support_for_ci is not None and support < min_support_for_ci:
        ci_reasons.append("insufficient_bootstrap_support")
    exact = _exact_cluster_distribution(keys, clusters, compute_delta) if len(keys) <= 10 else None
    return {
        "estimate": point["delta"],
        "recall_primary": point["recall_primary"],
        "recall_baseline": point["recall_baseline"],
        "ci_low": None if ci_reasons else lo,
        "ci_high": None if ci_reasons else hi,
        "ci_status": "not_estimable" if ci_reasons else "estimable",
        "ci_ineligibility_reasons": ci_reasons,
        "confidence_level": confidence_level,
        "replicates": replicates,
        "degenerate_replicates": nan_replicates,
        "cluster_key": cluster_key,
        "n_clusters": len(keys),
        "n_positive_clusters": positive_clusters,
        "n_rows": len(rows),
        "bootstrap_support_size": support,
        "degenerate_fraction": degenerate_fraction,
        "exact_cluster_distribution": exact,
        "seed": seed,
        "method": ("percentile_cluster_bootstrap_frozen_thresholds"
                   if frozen_thresholds is not None
                   else "percentile_cluster_bootstrap_recalibrated"),
        "frozen_thresholds": list(frozen_thresholds) if frozen_thresholds else None,
    }


def _exact_cluster_distribution(keys: list[str], clusters: dict[str, list[dict[str, Any]]],
                                compute_delta) -> dict[str, Any]:
    """Exact multinomial cluster-bootstrap distribution for K <= 10."""
    k = len(keys)
    total_sequences = k ** k
    mass: Counter[str] = Counter()
    degenerate_mass = 0.0
    compositions = 0
    for combo in combinations_with_replacement(range(k), k):
        counts = Counter(combo)
        multiplicity = factorial(k)
        sample: list[dict[str, Any]] = []
        for index, count in counts.items():
            multiplicity //= factorial(count)
            for _ in range(count):
                sample.extend(clusters[keys[index]])
        probability = multiplicity / total_sequences
        stat = compute_delta(sample)["delta"]
        compositions += 1
        if stat is None:
            degenerate_mass += probability
        else:
            mass[format(float(stat), ".12g")] += probability
    return {
        "compositions": compositions,
        "degenerate_probability": degenerate_mass,
        "support": [
            {"value": float(value), "probability": probability}
            for value, probability in sorted(mass.items(), key=lambda item: float(item[0]))
        ],
    }


def cluster_bootstrap_statistic(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    cluster_key: str = "template_id",
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20261019,
    min_total_clusters_for_ci: int | None = None,
    min_positive_clusters_for_ci: int | None = None,
    max_degenerate_fraction_for_ci: float | None = None,
    min_support_for_ci: int | None = None,
) -> dict[str, Any]:
    """Generic cluster bootstrap for any scalar statistic (e.g. AUROC delta)."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(row[cluster_key], []).append(row)
    keys = sorted(clusters)
    if not keys:
        return {"estimate": None, "ci_low": None, "ci_high": None, "error": "no_data"}
    point = statistic(rows)
    rng = random.Random(seed)
    values: list[float] = []
    degenerate = 0
    for _ in range(replicates):
        sample: list[dict[str, Any]] = []
        for _ in range(len(keys)):
            sample.extend(clusters[rng.choice(keys)])
        try:
            value = statistic(sample)
        except Exception:
            degenerate += 1
            continue
        if value is None or value != value:  # NaN guard
            degenerate += 1
            continue
        values.append(value)
    if not values:
        return {"estimate": point, "ci_low": None, "ci_high": None,
                "error": "all_replicates_degenerate"}
    values.sort()
    alpha = 1.0 - confidence_level
    positive_clusters = len({r[cluster_key] for r in rows if r["label"] == 1})
    degenerate_fraction = degenerate / replicates
    support = len(set(values))
    ci_reasons: list[str] = []
    if min_total_clusters_for_ci is not None and len(keys) < min_total_clusters_for_ci:
        ci_reasons.append("too_few_total_clusters")
    if (min_positive_clusters_for_ci is not None
            and positive_clusters < min_positive_clusters_for_ci):
        ci_reasons.append("too_few_positive_clusters")
    if (max_degenerate_fraction_for_ci is not None
            and degenerate_fraction > max_degenerate_fraction_for_ci):
        ci_reasons.append("too_many_degenerate_replicates")
    if min_support_for_ci is not None and support < min_support_for_ci:
        ci_reasons.append("insufficient_bootstrap_support")
    return {
        "estimate": point,
        "ci_low": (None if ci_reasons else values[int(alpha / 2 * len(values))]),
        "ci_high": (None if ci_reasons else
                    values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]),
        "ci_status": "not_estimable" if ci_reasons else "estimable",
        "ci_ineligibility_reasons": ci_reasons,
        "confidence_level": confidence_level,
        "replicates": replicates,
        "degenerate_replicates": degenerate,
        "cluster_key": cluster_key,
        "n_clusters": len(keys),
        "n_positive_clusters": positive_clusters,
        "bootstrap_support_size": support,
        "degenerate_fraction": degenerate_fraction,
        "seed": seed,
        "method": "percentile_cluster_bootstrap",
    }


__all__ = ["cluster_bootstrap_paired", "cluster_bootstrap_statistic"]
