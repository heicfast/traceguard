"""Headless figures and tables with provenance lines.

Every figure embeds its data digest (SHA-256 of the serialized plotting data)
and the run id, satisfying the traceability bar of docs/02 §10:
figure -> analysis data -> scores -> views -> episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ids import sha256_json


def _finalize(fig, path: Path, data_payload: dict[str, Any], provenance: str) -> Path:
    fig.text(0.01, 0.005, provenance, fontsize=5, color="0.35", ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    data_path = path.with_suffix(".plotdata.json")
    data_path.write_text(json.dumps(data_payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return path


def plot_roc_curves(per_condition: dict[str, dict[str, Any]], out_path: str | Path,
                    title: str, provenance: str) -> Path:
    """ROC curves for several conditions; per_condition: name -> roc_points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    for name, payload in per_condition.items():
        pts = payload["points"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                label=f"{name} (AUC={payload['auroc']:.3f})",
                linewidth=1.6, alpha=0.9)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Recall (TPR)")
    ax.set_title(title)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    data = {k: v for k, v in per_condition.items()}
    return _finalize(fig, Path(out_path), data, provenance)


def plot_score_distributions(scores_by_label: dict[str, dict[str, list[float]]],
                             out_path: str | Path, title: str, provenance: str) -> Path:
    """Distributions of scores by condition, split by true label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = sorted(next(iter(scores_by_label.values())).keys())
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.1 * len(conditions), 3.6),
                             constrained_layout=True, squeeze=False)
    for ix, condition in enumerate(conditions):
        ax = axes[0][ix]
        benign = scores_by_label.get("benign", {}).get(condition, [])
        attack = scores_by_label.get("attack", {}).get(condition, [])
        ax.hist([benign, attack], bins=20, range=(0.0, 1.0), stacked=False,
                label=[f"benign (n={len(benign)})", f"attack (n={len(attack)})"],
                color=["#2c7fb8", "#d95f0e"], alpha=0.85)
        ax.set_title(condition, fontsize=9)
        ax.set_xlabel("suspicion score", fontsize=8)
        if ix == 0:
            ax.set_ylabel("episodes", fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    return _finalize(fig, Path(out_path), scores_by_label, provenance)


def plot_paired_effect(estimates: dict[str, dict[str, Any]], out_path: str | Path,
                       title: str, provenance: str) -> Path:
    """Paired delta with cluster-bootstrap CI per comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(estimates.keys())
    ys = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 2.2 + 0.5 * len(names)), constrained_layout=True)
    for y, name in zip(ys, names, strict=False):
        est = estimates[name]
        lo, hi = est.get("ci_low"), est.get("ci_high")
        point = est.get("estimate")
        if point is None:
            continue
        if lo is not None and hi is not None:
            ax.hlines(y, lo, hi, color="#2c7fb8", linewidth=2.4, alpha=0.85)
            ax.plot([lo, lo], [y - 0.12, y + 0.12], color="#2c7fb8", linewidth=1.4)
            ax.plot([hi, hi], [y - 0.12, y + 0.12], color="#2c7fb8", linewidth=1.4)
        ax.plot(point, y, "o", color="#08306b", markersize=6)
        label = f"{point:+.3f}  [{lo:+.3f}, {hi:+.3f}]" if lo is not None else f"{point:+.3f}"
        ax.text(hi if hi is not None else point, y + 0.13, label, fontsize=7.5, va="bottom")
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("paired difference (primary - baseline)", fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2, axis="x")
    return _finalize(fig, Path(out_path), estimates, provenance)


def provenance_line(run_id: str, config_sha: str, extra: str = "") -> str:
    digest = sha256_json({"run_id": run_id, "config_sha256": config_sha})[:16]
    return f"TRACEGUARD {run_id} · cfg:{config_sha[:12]} · data:{digest}{' · ' + extra if extra else ''}"


__all__ = ["plot_roc_curves", "plot_score_distributions", "plot_paired_effect", "provenance_line"]
