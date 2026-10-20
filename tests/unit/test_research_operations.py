from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from traceguard.analyses.metrics import (
    exact_mcnemar,
    holm_adjust,
    precision_recall_points,
    threshold_at_fpr,
)
from traceguard.provenance import build_evidence_index, verify_evidence_index
from traceguard.study_design import PlanningAssumptions, paired_binary_sample_size
from traceguard.traces.storage import atomic_write_json
from traceguard.workflow import WorkflowJournal, run_workflow


def test_exact_mcnemar_and_holm_are_hand_checkable():
    result = exact_mcnemar([True, True, True, False], [False, False, True, False])
    assert result["primary_only"] == 2
    assert result["baseline_only"] == 0
    assert result["p_value_two_sided"] == pytest.approx(0.5)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_precision_recall_groups_ties():
    points = precision_recall_points([0.9, 0.5], [0.5, 0.1])
    assert points[0] == pytest.approx((0.5, 1.0, 0.9))
    assert points[1] == pytest.approx((1.0, 2 / 3, 0.5))


def test_threshold_nextafter_handles_negative_zero():
    threshold = threshold_at_fpr([-0.0, -0.0], 0.0)
    assert threshold > 0.0


def test_study_size_increases_with_cluster_correlation():
    independent = paired_binary_sample_size(PlanningAssumptions(intracluster_correlation=0.01))
    clustered = paired_binary_sample_size(PlanningAssumptions(intracluster_correlation=0.30))
    assert (
        clustered["cluster_adjusted_attempted_positive_pairs"]
        > independent["cluster_adjusted_attempted_positive_pairs"]
    )


def test_evidence_index_detects_mutation(tmp_path: Path):
    atomic_write_json(
        tmp_path / "run_manifest.json",
        {
            "run_id": "test",
            "experiment_stage": "smoke",
            "config_sha256": "a" * 64,
        },
    )
    atomic_write_json(tmp_path / "reports" / "result.json", {"value": 1})
    build_evidence_index(tmp_path)
    assert verify_evidence_index(tmp_path)["ok"]
    atomic_write_json(tmp_path / "reports" / "result.json", {"value": 2})
    assert not verify_evidence_index(tmp_path)["ok"]


def test_workflow_resume_verifies_output_digest(tmp_path: Path):
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema_version: 1.0.0\n"
        "stages:\n"
        "  - name: make\n"
        f"    command: [{sys.executable!r}, -c, \"from pathlib import Path; Path('result.txt').write_text('ok')\"]\n"
        "    outputs: [result.txt]\n",
        encoding="utf-8",
    )
    first = run_workflow(workflow, tmp_path / "work")
    second = run_workflow(workflow, tmp_path / "work")
    assert first["ok"] and second["ok"]
    assert second["plan"] == [{"stage": "make", "action": "skip_verified"}]
    records = WorkflowJournal(tmp_path / "work").records()
    assert [record["event"] for record in records] == ["started", "succeeded"]


def test_workflow_rejects_broken_journal(tmp_path: Path):
    journal = WorkflowJournal(tmp_path)
    journal.append({"event": "started", "stage": "x"})
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    records[0]["stage"] = "tampered"
    journal.path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        journal.records()


def test_workflow_rejects_output_path_escape(tmp_path: Path):
    workflow = tmp_path / "escape.yaml"
    workflow.write_text(
        "schema_version: 1.0.0\nstages:\n"
        "  - name: bad\n    command: ['true']\n    outputs: [../outside]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside the workflow root"):
        run_workflow(workflow, tmp_path / "work", dry_run=True)
