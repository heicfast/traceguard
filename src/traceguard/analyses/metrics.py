"""Detection metrics: recall at calibrated FPR, ROC/PR curves, AUROC/AP.

All functions are pure and independently unit-testable against tiny
hand-computable fixtures (tests/properties/test_metrics_fixture.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def threshold_at_fpr(benign_scores: Sequence[float], fpr_target: float) -> float:
    """Choose the alarm threshold achieving FPR <= target on benign scores.

    Conservative tie handling: allow at most k = floor(target * n) false
    positives; the threshold sits just ABOVE the k-th largest benign score, so
    scores strictly tying with benign scores are not counted as detections.
    """
    n = len(benign_scores)
    if n == 0:
        raise ValueError("no benign scores: cannot calibrate a threshold")
    if not (0.0 <= fpr_target <= 1.0):
        raise ValueError("fpr_target out of range")
    k = int(math.floor(fpr_target * n + 1e-12))  # allowed false positives
    if k >= n:
        return -math.inf
    sorted_desc = sorted(benign_scores, reverse=True)
    # Threshold just above the k-th largest benign score: only scores strictly
    # greater than b[k] can be flagged, and they number at most k (the top k).
    # Ties with b[k] are excluded (conservative tie handling, documented).
    return _next_after(sorted_desc[k])


def threshold_at_fpr_inclusive_ties(
    benign_scores: Sequence[float],
    fpr_target: float,
) -> float:
    """Diagnostic threshold at the empirical boundary, including ties.

    This may exceed the target FPR when scores are coarse. It is reported only
    as a tie-sensitivity analysis; the conservative threshold remains primary.
    """
    n = len(benign_scores)
    if n == 0:
        raise ValueError("no benign scores: cannot calibrate a threshold")
    if not (0.0 <= fpr_target <= 1.0):
        raise ValueError("fpr_target out of range")
    k = int(math.floor(fpr_target * n + 1e-12))
    if k >= n:
        return -math.inf
    return sorted(benign_scores, reverse=True)[k]


def _next_after(x: float) -> float:
    """Smallest representable float strictly greater than ``x``."""
    return math.nextafter(x, math.inf)


def false_positive_rate(benign_scores: Sequence[float], threshold: float) -> float:
    if not benign_scores:
        return float("nan")
    return sum(1 for s in benign_scores if s >= threshold) / len(benign_scores)


def recall_at_threshold(attack_scores: Sequence[float], threshold: float) -> float:
    if not attack_scores:
        return float("nan")
    return sum(1 for s in attack_scores if s >= threshold) / len(attack_scores)


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Rank-based AUROC with exact tie handling (Mann-Whitney U)."""
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = [(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores]
    combined.sort(key=lambda pair: pair[0])
    ranks: list[float] = []
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        tie_size = j - i + 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks.extend([avg_rank] * tie_size)
        i = j + 1
    rank_sum_pos = sum(r for (_, lab), r in zip(combined, ranks, strict=False) if lab == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def average_precision(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Step-wise average precision with threshold/tie grouping (sklearn-compatible).

    AP = sum over distinct thresholds (descending) of
        precision_at_threshold * (recall_gain_at_threshold).
    Cross-checked against sklearn.metrics.average_precision_score in tests.
    """
    n_pos = len(positive_scores)
    if n_pos == 0:
        return float("nan")
    pairs = sorted(
        [(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores],
        key=lambda p: -p[0],
    )
    # group by distinct threshold values
    tp_cum = 0
    fp_cum = 0
    ap = 0.0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        group_tp = 0
        group_fp = 0
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1] == 1:
                group_tp += 1
            else:
                group_fp += 1
            i += 1
        tp_cum += group_tp
        fp_cum += group_fp
        if group_tp > 0:
            precision = tp_cum / (tp_cum + fp_cum)
            recall_gain = group_tp / n_pos
            ap += precision * recall_gain
    return ap


def roc_points(
    positive_scores: Sequence[float], negative_scores: Sequence[float], grid: int = 101
) -> list[tuple[float, float]]:
    """(FPR, TPR) points over a fixed threshold grid for stable figures."""
    pts: list[tuple[float, float]] = []
    values = sorted(set(list(positive_scores) + list(negative_scores)))
    thresholds = [-1e-9] + [v - 1e-9 for v in values] + [1.0 + 1e-9]
    for t in thresholds:
        fpr = false_positive_rate(negative_scores, t)
        tpr = recall_at_threshold(positive_scores, t)
        pts.append((fpr, tpr))
    pts.sort()
    return pts


def precision_recall_points(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> list[tuple[float, float, float]]:
    """Return ``(recall, precision, threshold)`` at every distinct score.

    Ties are admitted as a group. This is important for TRACEGUARD's often
    coarse monitor scores: breaking ties by record order would manufacture
    resolution that the scorer did not produce.
    """
    n_pos = len(positive_scores)
    if n_pos == 0:
        return []
    pairs = sorted(
        [(float(score), 1) for score in positive_scores]
        + [(float(score), 0) for score in negative_scores],
        key=lambda pair: -pair[0],
    )
    points: list[tuple[float, float, float]] = []
    tp = fp = index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            if pairs[index][1] == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        points.append((tp / n_pos, tp / (tp + fp), threshold))
    return points


def exact_mcnemar(
    primary_detected: Sequence[bool], baseline_detected: Sequence[bool]
) -> dict[str, float | int | str | None]:
    """Two-sided exact McNemar test for paired binary detections.

    The test conditions on discordant pairs and uses a Binomial(n, 0.5)
    reference distribution. It is an inferential companion to the paired
    recall difference, not a replacement for its cluster-aware interval.
    """
    if len(primary_detected) != len(baseline_detected):
        raise ValueError("paired detection vectors must have equal length")
    primary_only = sum(
        p and not b for p, b in zip(primary_detected, baseline_detected, strict=True)
    )
    baseline_only = sum(
        b and not p for p, b in zip(primary_detected, baseline_detected, strict=True)
    )
    discordant = primary_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(0, min(primary_only, baseline_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    n = len(primary_detected)
    return {
        "n_pairs": n,
        "primary_only": primary_only,
        "baseline_only": baseline_only,
        "discordant": discordant,
        "recall_difference": ((primary_only - baseline_only) / n if n else None),
        "p_value_two_sided": p_value,
        "method": "exact_mcnemar_conditional_binomial",
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise-error adjustment, returned in original order."""
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 0.0
    count = len(ordered)
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[original_index] = running
    return adjusted


__all__ = [
    "threshold_at_fpr",
    "threshold_at_fpr_inclusive_ties",
    "false_positive_rate",
    "recall_at_threshold",
    "auroc",
    "average_precision",
    "roc_points",
    "precision_recall_points",
    "exact_mcnemar",
    "holm_adjust",
]
