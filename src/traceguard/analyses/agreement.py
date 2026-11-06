"""Inter-monitor agreement (Cohen's kappa) and confusion accounting.

When multiple monitors score the SAME episodes (the matrix design), their
detection decisions can be cross-tabulated.  Raw percent agreement is
inflated by base rates; Cohen's kappa (Cohen 1960) corrects for chance.
Kappa on binary flag decisions is reported per (condition, threshold) pair,
always with the raw confusion matrix so nothing is hidden behind the
coefficient.  This measures monitor ENSEMBLE diversity — it is never a
ground-truth substitute (rule 2).
"""

from __future__ import annotations

from typing import Any


def confusion_matrix(rater_a: list[bool], rater_b: list[bool]) -> dict[str, int]:
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("rater vectors must be non-empty and equal length")
    return {
        "both_flag": sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a and b),
        "a_only": sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a and not b),
        "b_only": sum(1 for a, b in zip(rater_a, rater_b, strict=True) if b and not a),
        "neither": sum(1 for a, b in zip(rater_a, rater_b, strict=True) if not a and not b),
    }


def cohen_kappa(rater_a: list[bool], rater_b: list[bool]) -> dict[str, Any]:
    """Cohen's kappa with the 2x2 confusion counts."""
    counts = confusion_matrix(rater_a, rater_b)
    n = sum(counts.values())
    if n == 0:
        return {"kappa": None, "error": "no_data", **counts}
    po = (counts["both_flag"] + counts["neither"]) / n
    pa = (counts["both_flag"] + counts["a_only"]) / n
    pb = (counts["both_flag"] + counts["b_only"]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if pe >= 1.0 else (po - pe) / (1 - pe)
    return {
        "kappa": kappa,
        "percent_agreement": po,
        "expected_agreement": pe,
        "n": n,
        **counts,
    }


def agreement_matrix(
    decisions: dict[str, list[bool]],
) -> dict[str, Any]:
    """Pairwise kappa over named monitor decision vectors (equal lengths)."""
    names = sorted(decisions)
    lengths = {len(decisions[name]) for name in names}
    if len(lengths) > 1:
        raise ValueError("all monitor decision vectors must have equal length")
    pairwise: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f"{a}|{b}"] = cohen_kappa(decisions[a], decisions[b])
    return {"monitors": names, "n_episodes": lengths.pop() if lengths else 0,
            "pairwise": pairwise}


__all__ = ["confusion_matrix", "cohen_kappa", "agreement_matrix"]
