"""Calibration and decision-utility metrics for monitor scores.

Rank metrics (AUROC/AP) ignore whether scores behave like probabilities at
the operating point; a monitor deployed behind a threshold needs calibration
(Brier 1950; reliability diagrams) and decision utility (net benefit,
Vickers et al. 2006/2008).  All estimators are pure and hand-testable.
"""

from __future__ import annotations

from typing import Any


def brier_score(scores: list[float], labels: list[int]) -> float:
    """Mean squared error of scores (as probabilities) against binary labels."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and equal length")
    if any(not 0.0 <= s <= 1.0 for s in scores):
        raise ValueError("brier score requires scores in [0, 1]")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be binary")
    return sum((s - y) ** 2 for s, y in zip(scores, labels, strict=True)) / len(scores)


def reliability_bins(
    scores: list[float], labels: list[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width reliability bins: mean score, fraction positive, count."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and equal length")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for score, label in zip(scores, labels, strict=True):
        index = min(n_bins - 1, int(score * n_bins))
        bins[index].append((score, label))
    out: list[dict[str, float]] = []
    for index, members in enumerate(bins):
        if not members:
            out.append({"bin": index, "count": 0, "mean_score": float("nan"),
                        "fraction_positive": float("nan")})
            continue
        out.append({
            "bin": index,
            "count": len(members),
            "mean_score": sum(m[0] for m in members) / len(members),
            "fraction_positive": sum(m[1] for m in members) / len(members),
        })
    return out


def expected_calibration_error(scores: list[float], labels: list[int],
                               n_bins: int = 10) -> float:
    """ECE: count-weighted |mean score - fraction positive| over equal-width bins."""
    bins = reliability_bins(scores, labels, n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return float("nan")
    return sum(
        (b["count"] / total) * abs(b["mean_score"] - b["fraction_positive"])
        for b in bins if b["count"] > 0
    )


def net_benefit(
    scores: list[float], labels: list[int], threshold: float,
    *, action_value_when_positive: float = 1.0, action_cost_when_negative: float = 1.0,
) -> dict[str, float]:
    """Decision-curve net benefit of flagging at ``threshold``.

    ``action_value_when_positive`` / ``action_cost_when_negative`` encode the
    relative utility of a true flag vs a false flag (default 1:1, i.e. the
    classic Vickers convention NB = TP/n - FP/n * (pt/(1-pt))).  Utilities
    must be set BEFORE seeing results (predeclared in the frozen config).
    """
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and equal length")
    if action_value_when_positive <= 0 or action_cost_when_negative <= 0:
        raise ValueError("utilities must be positive")
    n = len(labels)
    flagged = [s >= threshold for s in scores]
    tp = sum(1 for flag, y in zip(flagged, labels, strict=True) if flag and y == 1)
    fp = sum(1 for flag, y in zip(flagged, labels, strict=True) if flag and y == 0)
    nb = (tp * action_value_when_positive - fp * action_cost_when_negative) / n
    treat_all = (sum(labels) * action_value_when_positive
                 - (n - sum(labels)) * action_cost_when_negative) / n
    return {
        "threshold": threshold,
        "net_benefit": nb,
        "net_benefit_flag_all": treat_all,
        "net_benefit_flag_none": 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "n": n,
    }


def calibration_report(scores: list[float], labels: list[int],
                       threshold: float) -> dict[str, Any]:
    return {
        "brier_score": brier_score(scores, labels),
        "expected_calibration_error_10": expected_calibration_error(scores, labels, 10),
        "reliability_bins": reliability_bins(scores, labels, 10),
        "net_benefit": net_benefit(scores, labels, threshold),
    }


__all__ = [
    "brier_score",
    "reliability_bins",
    "expected_calibration_error",
    "net_benefit",
    "calibration_report",
]
