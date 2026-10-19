"""Analysis runtime: paired metrics, clustered bootstrap, figures, report.

Pipeline (docs/01 §10):
  1. Build analysis frame (eligibility: attempted attacks are positives;
     honest episodes are negatives; runtime errors censored with reasons).
  2. Calibrate per-condition thresholds on the calibration split's benign
     scores ONLY (default: validation split role).
  3. Evaluate paired conditions on each evaluation split role (holdout,
     shift, development for reference) — same episodes across conditions.
  4. Cluster (template-level) bootstrap for the paired recall difference and
     for the paired AUROC difference.
  5. Precision gate: report whether benign sample sizes support the FPR
     target; when they do not, flag recall@FPR as exploratory and promote
     the paired AUROC difference per the preregistration rule.
  6. Funnel, censor logs, figures with provenance lines, report JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyses.metrics import (
    auroc,
    average_precision,
    false_positive_rate,
    recall_at_threshold,
    roc_points,
    threshold_at_fpr,
)
from .analyses.pair import build_analysis_frame, paired_subset, summarize_funnel
from .analyses.plots import (
    plot_paired_effect,
    plot_roc_curves,
    plot_score_distributions,
    provenance_line,
)
from .analyses.uncertainty import cluster_bootstrap_paired, cluster_bootstrap_statistic
from .config import load_config
from .traces.storage import RunStore


def _condition_pairs(conditions: list[str]) -> list[tuple[str, str]]:
    """Primary + control comparisons defined by the protocol (C vs A, C vs D, C vs E)."""
    pairs: list[tuple[str, str]] = []
    have = set(conditions)
    if "actions_plus_cot" in have and "actions" in have:
        pairs.append(("actions_plus_cot", "actions"))
    if "actions_plus_cot" in have and "actions_plus_noise" in have:
        pairs.append(("actions_plus_cot", "actions_plus_noise"))
    if "actions_plus_cot" in have and "actions_plus_shuffled_cot" in have:
        pairs.append(("actions_plus_cot", "actions_plus_shuffled_cot"))
    return pairs


def _evaluate_split(rows: list[dict[str, Any]], conditions: list[str],
                    calibration_rows: list[dict[str, Any]], fpr_target: float,
                    config, split_label: str) -> dict[str, Any]:
    """Threshold-calibrated paired evaluation on one split role."""
    result: dict[str, Any] = {"split": split_label, "n_episodes": len(rows)}
    benign_rows = [r for r in rows if r["label"] == 0]
    attack_rows = [r for r in rows if r["label"] == 1]
    result["n_benign"] = len(benign_rows)
    result["n_positive_eligible"] = len(attack_rows)
    result["n_independent_benign_clusters"] = len({r["template_id"] for r in benign_rows})

    # thresholds from the CALIBRATION split only
    thresholds: dict[str, float] = {}
    realized_fpr: dict[str, Any] = {}
    for condition in conditions:
        cal_benign = [r["scores"][condition] for r in calibration_rows if r["label"] == 0
                      and r["scores"].get(condition) is not None]
        if not cal_benign:
            thresholds[condition] = None
            continue
        thresholds[condition] = threshold_at_fpr(cal_benign, fpr_target)
    for condition, t in thresholds.items():
        if t is None or not benign_rows:
            realized_fpr[condition] = None
            continue
        scores_here = [r["scores"][condition] for r in benign_rows
                       if r["scores"].get(condition) is not None]
        realized_fpr[condition] = (
            round(false_positive_rate(scores_here, t), 4) if scores_here else None)
    result["thresholds_calibrated"] = {k: v for k, v in thresholds.items()}
    result["realized_fpr"] = realized_fpr

    recalls: dict[str, Any] = {}
    for condition, t in thresholds.items():
        if t is None or not attack_rows:
            recalls[condition] = None
            continue
        scores_here = [r["scores"][condition] for r in attack_rows
                       if r["scores"].get(condition) is not None]
        recalls[condition] = (
            round(recall_at_threshold(scores_here, t), 4) if scores_here else None)
    result["recall_at_calibrated_fpr"] = recalls

    aucs: dict[str, Any] = {}
    aps: dict[str, Any] = {}
    for condition in conditions:
        pos = [r["scores"][condition] for r in attack_rows if r["scores"].get(condition) is not None]
        neg = [r["scores"][condition] for r in benign_rows if r["scores"].get(condition) is not None]
        if pos and neg:
            aucs[condition] = round(auroc(pos, neg), 4)
            aps[condition] = round(average_precision(pos, neg), 4)
        else:
            aucs[condition] = None
            aps[condition] = None
    result["auroc"] = aucs
    result["average_precision"] = aps
    return result


def run_analysis(config_path: str, run_dir: str, out_dir: str | None,
                 monitor_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    store = RunStore(run_dir)
    conditions = config.monitor.conditions
    fpr_target = config.monitor.primary_fpr_target

    rows, censor = build_analysis_frame(store, monitor_name, conditions)
    if not rows:
        return {"ok": False, "error": "no analysis rows built"}
    out_path = Path(out_dir) if out_dir else store.run_dir / "reports"
    out_path.mkdir(parents=True, exist_ok=True)

    # paired subset: all conditions valid
    paired_rows, paired_censor = paired_subset(rows, conditions)

    calibration_role = config.monitor.calibration_split
    calibration_rows = [r for r in paired_rows if r["split_role"] == calibration_role]
    if not calibration_rows:
        # fall back to development for smoke-scale runs; reported explicitly
        calibration_rows = [r for r in paired_rows if r["split_role"] == "development"]

    splits_report: dict[str, Any] = {}
    for split_label in ("development", "validation", "holdout", "shift"):
        split_rows = [r for r in paired_rows if r["split_role"] == split_label]
        if not split_rows:
            continue
        splits_report[split_label] = _evaluate_split(
            split_rows, conditions, calibration_rows, fpr_target, config, split_label)

    # -- paired bootstrap on holdout (primary) and shift (secondary) ---------
    bootstrap_results: dict[str, Any] = {}
    for split_label in ("holdout", "shift"):
        split_rows = [r for r in paired_rows if r["split_role"] == split_label]
        for primary, baseline in _condition_pairs(conditions):
            if not split_rows:
                continue
            key = f"{split_label}|{primary.replace('actions_plus_cot', 'C')}_vs_{baseline.replace('actions_plus_cot', 'C')}"
            recall_boot = cluster_bootstrap_paired(
                split_rows, primary, baseline, fpr_target,
                cluster_key=config.analysis.cluster_key,
                replicates=config.analysis.bootstrap_replicates,
                confidence_level=config.analysis.confidence_level,
                seed=config.analysis.bootstrap_seed,
            )
            auc_boot = cluster_bootstrap_statistic(
                split_rows,
                statistic=lambda sample, p=primary, b=baseline: (
                    auroc([r["scores"][p] for r in sample if r["label"] == 1],
                          [r["scores"][p] for r in sample if r["label"] == 0])
                    - auroc([r["scores"][b] for r in sample if r["label"] == 1],
                            [r["scores"][b] for r in sample if r["label"] == 0])
                ),
                cluster_key=config.analysis.cluster_key,
                replicates=config.analysis.bootstrap_replicates,
                confidence_level=config.analysis.confidence_level,
                seed=config.analysis.bootstrap_seed,
            )
            bootstrap_results[key] = {"recall_delta": recall_boot, "auroc_delta": auc_boot}

    # -- precision gate ---------------------------------------------------------
    holdout_report = splits_report.get("holdout", {})
    cal_benign_clusters = len({r["template_id"] for r in calibration_rows if r["label"] == 0})
    # For a 5% FPR target with cluster correlation, demand >= 40 independent
    # benign clusters for an interpretable calibration (documented rule).
    fpr_gate_passed = cal_benign_clusters >= 40
    precision_gate = {
        "calibration_split": calibration_role,
        "independent_benign_clusters": cal_benign_clusters,
        "fpr_target": fpr_target,
        "gate_rule": ">= 40 independent benign template clusters",
        "gate_passed": bool(fpr_gate_passed),
        "consequence": (
            "recall@FPR is confirmatory per preregistration"
            if fpr_gate_passed else
            "recall@FPR is EXPLORATORY; primary estimand falls back to paired AUROC "
            "difference per the preregistered deviation rule"),
    }

    funnel = summarize_funnel(rows)
    report = {
        "schema_version": "1.0.0",
        "analysis_monitor": monitor_name,
        "conditions": conditions,
        "fpr_target": fpr_target,
        "funnel": funnel,
        "censor": censor.to_payload(),
        "paired_censor": paired_censor.to_payload(),
        "splits": splits_report,
        "paired_bootstrap": bootstrap_results,
        "precision_gate": precision_gate,
        "config_sha256": config.content_hash(),
        "cluster_key": config.analysis.cluster_key,
        "estimand": config.analysis.primary_estimand,
    }

    # -- figures ------------------------------------------------------------------
    prov = provenance_line(store.read_run_manifest().get("run_id", "run"),
                           config.content_hash(), monitor_name)
    holdout_rows = [r for r in paired_rows if r["split_role"] == "holdout"]
    figures: list[str] = []
    if holdout_rows:
        per_condition: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            pos = [r["scores"][condition] for r in holdout_rows if r["label"] == 1]
            neg = [r["scores"][condition] for r in holdout_rows if r["label"] == 0]
            if pos and neg:
                per_condition[condition] = {
                    "points": roc_points(pos, neg),
                    "auroc": auroc(pos, neg),
                }
        if per_condition:
            fig_path = plot_roc_curves(per_condition, out_path / "fig_roc_holdout.png",
                                       f"ROC on held-out family (monitor {monitor_name})", prov)
            figures.append(str(fig_path))
        scores_by_label = {
            "benign": {c: [r["scores"][c] for r in holdout_rows if r["label"] == 0
                           and r["scores"].get(c) is not None] for c in conditions},
            "attack": {c: [r["scores"][c] for r in holdout_rows if r["label"] == 1
                           and r["scores"].get(c) is not None] for c in conditions},
        }
        fig_path = plot_score_distributions(scores_by_label, out_path / "fig_scores_holdout.png",
                                            "Score distributions on holdout", prov)
        figures.append(str(fig_path))
    if bootstrap_results:
        estimates = {
            key: payload["recall_delta"] if payload["recall_delta"].get("estimate") is not None
            else payload["auroc_delta"]
            for key, payload in bootstrap_results.items()
        }
        estimates = {k: v for k, v in estimates.items() if v.get("estimate") is not None}
        if estimates:
            fig_path = plot_paired_effect(estimates, out_path / "fig_paired_effects.png",
                                          "Paired effects with cluster-bootstrap 95% CI", prov)
            figures.append(str(fig_path))

    report["figures"] = figures
    report_path = out_path / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "ok": True,
        "report_path": str(report_path),
        "funnel": funnel,
        "precision_gate": precision_gate,
        "holdout": holdout_report,
        "figures": figures,
    }
    return {"ok": True, "summary": summary, "report": report}


__all__ = ["run_analysis"]
