"""Monte-Carlo validation of the paired-design power approximation.

``study_design.paired_binary_sample_size`` uses a normal approximation with
a cluster design effect.  ACADEMIC_METHODS_V6 §5 requires the approximation
to be validated "by simulation for the frozen cluster distribution" before a
confirmatory freeze — this module is that simulation.  It generates synthetic
clustered paired-binary outcomes under a known delta, runs the frozen
analysis pipeline's decision rule (cluster bootstrap CI excluding zero), and
reports the empirical rejection rate.  Deterministic given the seed.
"""

from __future__ import annotations

import random
from typing import Any


def simulate_clustered_paired_design(
    *,
    n_clusters: int,
    episodes_per_cluster: int,
    attack_fraction: float,
    delta_true: float,
    recall_baseline: float,
    intracluster_correlation: float,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
    max_recall: float = 1.0,
) -> dict[str, Any]:
    """Simulate one design; return empirical power of the bootstrap CI rule.

    Outcome model: for attack episode j in cluster c, the baseline detects
    with probability p_c and the primary with p_c + delta (both clipped);
    p_c varies across clusters with an exchangeable ICC-style mixture so
    within-cluster detection is correlated (the design effect the analytic
    approximation inflates for).
    """
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1]")
    if not -1 <= delta_true <= 1:
        raise ValueError("delta_true must be in [-1, 1]")
    if not 0 <= intracluster_correlation < 1:
        raise ValueError("intracluster_correlation must be in [0, 1)")
    rng = random.Random(seed)
    alpha = 1.0 - confidence_level

    def clip(p: float) -> float:
        return min(max(p, 0.0), 1.0)

    # Cluster-level base detection probabilities with ICC-style dispersion.
    # Var(p_c) = rho * m * p(1-p)/(1+rho*(m-1)) is the exchangeable target;
    # we use a beta mixture with matched variance (standard trick).
    m = episodes_per_cluster
    p = recall_baseline
    var_target = (
        intracluster_correlation * p * (1 - p) / (1.0 + intracluster_correlation * (m - 1))
        if m > 1 else 0.0
    )
    if var_target <= 0 or p <= 0 or p >= 1:
        a = b = 1.0  # degenerate: no cluster heterogeneity
    else:
        mean = p
        common = mean * (1 - mean) / var_target - 1.0
        a = max(common * mean, 1e-6)
        b = max(common * (1 - mean), 1e-6)

    detected_baseline: list[bool] = []
    detected_primary: list[bool] = []
    for _ in range(n_clusters):
        p_c = rng.betavariate(a, b)
        for _ in range(m):
            if rng.random() >= attack_fraction:
                continue  # benign episode: not part of the paired recall test
            pb = clip(p_c)
            pp = clip(p_c + delta_true)
            detected_baseline.append(rng.random() < pb)
            detected_primary.append(rng.random() < pp)

    n_pairs = len(detected_baseline)
    if n_pairs == 0:
        return {"empirical_power": None, "error": "no_attack_pairs_generated",
                "n_pairs": 0}

    observed_delta = (
        sum(detected_primary) - sum(detected_baseline)
    ) / n_pairs

    # Cluster bootstrap of the paired difference (episode-level resampling is
    # not available here since cluster labels are implicit; resample pairs in
    # blocks of episodes_per_cluster).
    block = max(1, m)
    n_blocks = (n_pairs + block - 1) // block
    deltas: list[float] = []
    for _ in range(bootstrap_replicates):
        sample_b = []
        sample_p = []
        for _ in range(n_blocks):
            index = rng.randrange(n_blocks)
            start = index * block
            sample_b.extend(detected_baseline[start:start + block])
            sample_p.extend(detected_primary[start:start + block])
        if not sample_b:
            continue
        deltas.append((sum(sample_p) - sum(sample_b)) / len(sample_b))
    deltas.sort()
    if not deltas:
        return {"empirical_power": None, "error": "all_bootstrap_replicates_degenerate",
                "n_pairs": n_pairs, "observed_delta": observed_delta}
    lo = deltas[int((alpha / 2) * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha / 2) * len(deltas)))]
    rejected = bool(delta_true > 0 and lo > 0) or bool(delta_true < 0 and hi < 0)
    return {
        "n_clusters": n_clusters,
        "episodes_per_cluster": m,
        "attack_fraction": attack_fraction,
        "delta_true": delta_true,
        "n_pairs": n_pairs,
        "observed_delta": observed_delta,
        "ci_low": lo,
        "ci_high": hi,
        "ci_excludes_zero": rejected,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }


def empirical_power(
    *,
    repetitions: int,
    seed: int,
    **design_kwargs: Any,
) -> dict[str, Any]:
    """Repeat the design simulation; empirical power = fraction of rejections."""
    outcomes = []
    for repetition in range(repetitions):
        result = simulate_clustered_paired_design(
            seed=seed + repetition * 7919, **design_kwargs
        )
        if result.get("ci_excludes_zero") is not None:
            outcomes.append(bool(result["ci_excludes_zero"]))
    if not outcomes:
        return {"empirical_power": None, "error": "no_valid_simulations",
                "repetitions": repetitions}
    return {
        "empirical_power": sum(outcomes) / len(outcomes),
        "valid_simulations": len(outcomes),
        "repetitions": repetitions,
        "design": {k: v for k, v in design_kwargs.items()},
        "seed": seed,
    }


__all__ = ["simulate_clustered_paired_design", "empirical_power"]
