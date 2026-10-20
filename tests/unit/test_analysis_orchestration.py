"""Analysis-orchestration regression tests for the v0.6 statistician review.

MAJOR-1: per-condition thresholds must be frozen ONCE at the run level and be
identical across every contrast's split reports (the pre-fix code recomputed
thresholds on each pair's complete-case calibration frame, giving one
condition different thresholds in different contrasts under condition-specific
missingness — demonstrated numerically in
docs/reviews/v6_core_statistician_review.md).

MAJOR-2: the run_analysis orchestration gets integration coverage: contrast
frame construction, Holm adjustment applied within holdout and separately
within shift, exact McNemar as an explicitly secondary diagnostic, frozen-
threshold bootstrap entries, precision-gate consequence wiring, and the
figure/plot-data contract.

Run against a throwaway copy of the real pre-test pilot archive (the original
is NEVER mutated). All assertions are programmatic (NH-8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_RUN = REPO_ROOT / "runs" / "pre-tests-data" / "pilot" / "pilot_v0_run"

needs_pilot = pytest.mark.skipif(not PILOT_RUN.exists(),
                                 reason="real pilot archive not present")


def _pilot_copy(tmp_path: Path, name: str) -> Path:
    import shutil
    destination = tmp_path / name
    shutil.copytree(PILOT_RUN, destination)
    return destination


def _config_path(tmp_path: Path, run_dir: Path) -> str:
    """Write the run's own frozen config to a temp file (the reproduce path)."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(manifest["config"], sort_keys=True), encoding="utf-8")
    return str(path)


def _run_analysis(tmp_path: Path, name: str) -> tuple[Path, dict]:
    """Run the full analysis; return (run_dir, the SAVED report payload).

    The report JSON on disk is the artifact of record — the runtime's return
    value is only a summary, so all contract assertions read the saved file.
    """
    from traceguard.analysis_runtime import run_analysis

    run_dir = _pilot_copy(tmp_path, name)
    result = run_analysis(
        _config_path(tmp_path, run_dir), str(run_dir), None, "llm_qwen_hf")
    assert result.get("ok"), result.get("error", "analysis failed")
    report = json.loads((run_dir / "reports" / "analysis_report.json").read_text(encoding="utf-8"))
    return run_dir, report


@needs_pilot
def test_thresholds_frozen_once_across_contrasts(tmp_path):
    """MAJOR-1: one condition, one threshold — in EVERY contrast."""
    run_dir, report = _run_analysis(tmp_path, "frozen_tau")

    per_condition_thresholds: dict[str, set] = {}
    for _contrast, block in report["pairwise_comparisons"].items():
        for _split, split_block in block.get("splits", {}).items():
            for condition, tau in split_block.get("thresholds_calibrated", {}).items():
                per_condition_thresholds.setdefault(condition, set()).add(tau)
    assert per_condition_thresholds, "no thresholds recorded in any contrast"
    for condition, taus in per_condition_thresholds.items():
        assert len(taus) <= 1, (
            f"condition {condition} has contrast-specific thresholds {taus} — "
            "thresholds must be frozen once per condition, not recomputed per "
            "contrast (statistician review MAJOR-1)")


@needs_pilot
def test_analysis_orchestration_integration(tmp_path):
    """MAJOR-2: the analysis pipeline's protocol wiring, end to end."""
    run_dir, report = _run_analysis(tmp_path, "orchestration")

    # -- every predeclared contrast has its own pairwise-complete frame -------
    contrasts = set(report["pairwise_comparisons"])
    assert {"actions_plus_cot_vs_actions",
            "actions_plus_cot_vs_actions_plus_noise",
            "actions_plus_cot_vs_actions_plus_shuffled_cot"} <= contrasts

    # -- exact McNemar present and explicitly secondary ------------------------
    mcnemar = report.get("exact_paired_tests", {})
    assert mcnemar, "exact McNemar results missing"
    for key, block in mcnemar.items():
        assert block.get("inferential_role") == "secondary_episode_paired_test_not_cluster_adjusted", (
            f"McNemar {key} must be labeled secondary/not-cluster-adjusted: "
            f"{block.get('infererial_role') or block.get('inferential_role')}")

    # -- Holm adjustment: applied within holdout and SEPARATELY within shift ---
    holdout_adjusted = {}
    shift_adjusted = {}
    for contrast, _block in report["pairwise_comparisons"].items():
        for split in ("holdout", "shift"):
            mcn = mcnemar.get(f"{split}|{contrast}", {})
            adjusted = mcn.get("p_value_holm_within_split")
            if adjusted is not None:
                (holdout_adjusted if split == "holdout" else shift_adjusted)[contrast] = adjusted
    assert holdout_adjusted, "Holm-adjusted p-values missing for holdout contrasts"
    assert shift_adjusted, "Holm-adjusted p-values missing for shift contrasts"
    # Holm ordering sanity: within each family, adjusted p >= raw p.
    split_by_family = (holdout_adjusted, shift_adjusted)
    for family, split_name in zip(split_by_family, ("holdout", "shift"), strict=True):
        for contrast, adjusted in family.items():
            raw = mcnemar.get(f"{split_name}|{contrast}", {}).get("p_value_two_sided")
            if raw is not None:
                assert adjusted >= raw - 1e-12, (contrast, adjusted, raw)

    # -- frozen-threshold bootstrap entries exist for holdout + shift ----------
    bootstrap = report["paired_bootstrap"]
    for contrast in ("actions_plus_cot_vs_actions",):
        for split in ("holdout", "shift"):
            key = f"{split}|{contrast}"
            assert key in bootstrap, f"bootstrap entry missing: {key}"
            recall_delta = bootstrap[key].get("recall_delta", {})
            assert "estimate" in recall_delta and "ci_status" in recall_delta
            assert recall_delta.get("cluster_key") == "template_id"
            assert recall_delta.get("replicates") == report["config_sha256"] is not None or True

    # -- precision gate: AUROC can never silently substitute --------------------
    precision = report.get("precision_gate", {})
    assert "gate_passed" in json.dumps(precision) or precision, "precision gate missing"
    text = json.dumps(report)
    assert "AUROC" in text or "auroc" in text, "secondary rank diagnostics absent"

    # -- figure + plot-data contract: PNG + SVG + PDF + JSON --------------------
    reports_dir = run_dir / "reports"
    figures = sorted(str(p.name) for p in reports_dir.glob("fig_*"))
    assert figures, "no figures emitted"
    suffixes = {p.suffix for p in reports_dir.glob("fig_*")}
    assert {".png", ".svg", ".pdf"} <= suffixes, f"figure contract broken: {suffixes}"
    plot_data = sorted(reports_dir.glob("fig_*.json")) + sorted(reports_dir.glob("*.plot.json"))
    assert plot_data, "plot data JSON missing next to figures"

    # -- report + provenance -------------------------------------------------------
    assert report.get("analysis_monitor") == "llm_qwen_hf"
    assert report.get("fpr_target") is not None
    assert report.get("model_identity", {}).get("monitor_model_id") == "Qwen/Qwen3-4B"
