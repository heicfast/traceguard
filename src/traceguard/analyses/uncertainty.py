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
from collections.abc import Callable
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
) -> dict[str, Any]:
    """Bootstrap the paired recall difference at a calibrated FPR.

    Thresholds are recalibrated INSIDE every bootstrap replicate on its benign
    sample (only if the replicate contains benign rows; else that replicate
    contributes NaN and is dropped with an explicit count). Returns point
    estimate, percentile CI, and diagnostics.
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
        if not benign or not attacks:
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
    return {
        "estimate": point["delta"],
        "recall_primary": point["recall_primary"],
        "recall_baseline": point["recall_baseline"],
        "ci_low": lo,
        "ci_high": hi,
        "confidence_level": confidence_level,
        "replicates": replicates,
        "degenerate_replicates": nan_replicates,
        "cluster_key": cluster_key,
        "n_clusters": len(keys),
        "n_rows": len(rows),
        "seed": seed,
        "method": "percentile_cluster_bootstrap",
    }


def cluster_bootstrap_statistic(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    cluster_key: str = "template_id",
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20261019,
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
    return {
        "estimate": point,
        "ci_low": values[int(alpha / 2 * len(values))],
        "ci_high": values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))],
        "confidence_level": confidence_level,
        "replicates": replicates,
        "degenerate_replicates": degenerate,
        "cluster_key": cluster_key,
        "n_clusters": len(keys),
        "seed": seed,
        "method": "percentile_cluster_bootstrap",
    }


__all__ = ["cluster_bootstrap_paired", "cluster_bootstrap_statistic"]
