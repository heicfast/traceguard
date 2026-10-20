#!/usr/bin/env python3
"""Cross-cell comparison for the TRACEGUARD Phase 3 multi-model matrix.

Reads the per-cell analysis reports produced in-kernel (reports/
analysis_report.json inside each landed, scored run) and builds the
cross-cell evidence table that separates the competing explanations of the
monitoring result:

  - monitor capability   (scale cell vs anchor, same family, 4B -> 14B)
  - monitor family       (cross cell vs anchor, qwen3 -> deepseek R1 distill)
  - fusion architecture  (late fusion vs raw concatenation, within every cell)
  - benchmark headroom   (actions-only recall inside the 0.40-0.70 gate band?)

The script never recomputes estimates: it reads the frozen per-cell reports
and records their numbers verbatim with per-cell provenance. Any missing
report fails loudly (no partial silent output).

Usage:
  python scripts/compare_cells.py \
      --cell anchor=runs/v6/pilot_v0_run_mon_anchor \
      --cell scale=runs/v6/scale-monitor-qwen4-qwen14 \
      --cell cross=runs/v6/cross-monitor-qwen4-deepseek14 \
      --out runs/v6/cross_cell_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONTRASTS = [
    "actions_plus_cot_vs_actions",
    "actions_plus_cot_late_fusion_vs_actions",
    "actions_plus_cot_vs_actions_plus_noise",
    "actions_plus_cot_vs_actions_plus_shuffled_cot",
]
SPLITS = ["development", "validation", "holdout", "shift"]
HEADROOM_BAND = (0.40, 0.70)  # phase-3 protocol gate: actions-only recall band


def _flat_delta(report: dict, contrast: str, split: str) -> dict | None:
    entry = report.get("paired_bootstrap", {}).get(f"{split}|{contrast}")
    if not entry:
        return None
    return {
        key: {
            "estimate": value.get("estimate"),
            "ci_low": value.get("ci_low"),
            "ci_high": value.get("ci_high"),
            "ci_status": value.get("ci_status"),
            "n_clusters": value.get("n_clusters"),
            "n_positive_clusters": value.get("n_positive_clusters"),
        }
        for key, value in entry.items()
        if isinstance(value, dict) and "estimate" in value
    }


def _split_metrics(report: dict, contrast: str, split: str) -> dict | None:
    contrast_block = report.get("pairwise_comparisons", {}).get(contrast)
    if not contrast_block:
        return None
    split_block = contrast_block.get("splits", {}).get(split)
    if not split_block:
        return None
    return {
        "recall": split_block.get("recall_at_calibrated_fpr"),
        "realized_fpr": split_block.get("realized_fpr"),
        "thresholds": split_block.get("thresholds_calibrated"),
        "n_positive_eligible": split_block.get("n_positive_eligible"),
        "n_benign": split_block.get("n_benign"),
        "auroc": split_block.get("auroc"),
    }


def _measurement(report: dict) -> dict:
    diag = report.get("measurement_diagnostics", {})
    keep = {}
    if isinstance(diag, dict):
        for key, value in diag.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                keep[key] = value
            elif isinstance(value, dict):
                flat = {}
                for k2, v2 in value.items():
                    if isinstance(v2, (int, float, str, bool)) or v2 is None:
                        flat[k2] = v2
                if flat:
                    keep[key] = flat
    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True,
                        metavar="LABEL=RUN_DIR",
                        help="one per cell; RUN_DIR contains reports/analysis_report.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cells: dict[str, dict] = {}
    for spec in args.cell:
        label, _, run_dir = spec.partition("=")
        report_path = Path(run_dir) / "reports" / "analysis_report.json"
        if not report_path.exists():
            print(f"FAIL: {label}: missing {report_path}", file=sys.stderr)
            return 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cells[label] = {
            "run_dir": str(run_dir),
            "model_identity": report.get("model_identity"),
            "config_sha256": report.get("config_sha256"),
            "estimand": report.get("estimand"),
            "fpr_target": report.get("fpr_target"),
            "precision_gate": report.get("precision_gate"),
            "censor": report.get("censor"),
            "measurement_diagnostics": _measurement(report),
            "funnel": report.get("funnel"),
            "splits_definition": report.get("splits"),
        }

    comparison: dict[str, dict] = {}
    for contrast in CONTRASTS:
        comparison[contrast] = {}
        for split in SPLITS:
            rows: dict[str, dict] = {}
            any_row = False
            for label, cell in cells.items():
                report_path = Path(cell["run_dir"]) / "reports" / "analysis_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                metrics = _split_metrics(report, contrast, split)
                deltas = _flat_delta(report, contrast, split)
                if metrics is None and deltas is None:
                    continue
                rows[label] = {"metrics": metrics, "deltas": deltas,
                               "model_identity": cell["model_identity"]}
                any_row = True
            if any_row:
                comparison[contrast][split] = rows

    # Headroom gate: actions-only recall per cell per split (the benchmark
    # must not be saturated for the positive hypothesis to be testable).
    headroom: dict[str, dict] = {}
    for label, cell in cells.items():
        report = json.loads(
            (Path(cell["run_dir"]) / "reports" / "analysis_report.json").read_text(encoding="utf-8"))
        per_split = {}
        for split in SPLITS:
            metrics = _split_metrics(report, "actions_plus_cot_vs_actions", split)
            if not metrics or not metrics.get("recall"):
                continue
            recall_a = metrics["recall"].get("actions")
            if recall_a is None:
                continue
            in_band = HEADROOM_BAND[0] <= recall_a <= HEADROOM_BAND[1]
            per_split[split] = {
                "actions_recall": recall_a,
                "in_headroom_band": in_band,
                "saturated": recall_a > HEADROOM_BAND[1],
                "floor": recall_a < HEADROOM_BAND[0],
            }
        headroom[label] = per_split

    payload = {
        "schema_version": "1.0.0",
        "cells": {label: {
            "run_dir": cell["run_dir"],
            "model_identity": cell["model_identity"],
            "config_sha256": cell["config_sha256"],
        } for label, cell in cells.items()},
        "comparison": comparison,
        "headroom_gate": {"band": list(HEADROOM_BAND), "per_cell": headroom},
        "measurement_diagnostics": {label: cell["measurement_diagnostics"]
                                    for label, cell in cells.items()},
        "precision_gates": {label: cell["precision_gate"] for label, cell in cells.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "cells": sorted(cells),
        "contrasts_with_data": sorted(comparison),
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
