"""Regression tests for the v0.6.0 core developer-review findings F1-F9.

Each test names the review finding it guards, so a future edit that
reintroduces a defect fails with a self-explanatory message. The review
reproduced every finding by real execution (real signals, real journals,
byte-level surgery); these tests do the same. Per the review's display-hazard
discipline, no assertion relies on visually displayed bracketed content —
every check is programmatic (JSON parsing, byte-level reads, exit codes,
``os.kill`` liveness probes).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

import traceguard
from traceguard.cli import main as cli_main
from traceguard.errors import (
    GATE_FAILURE_EXIT_CODE,
    ErrorCode,
    classify_exception,
    classify_exit_code,
    retry_decision,
)
from traceguard.workflow import (
    JournalTamperError,
    LockHeldError,
    WorkflowBaselineError,
    WorkflowJournal,
    load_workflow,
    run_workflow,
    workflow_status,
)

SRC_ROOT = Path(traceguard.__file__).resolve().parents[1]


def write_workflow(path: Path, stages: list[dict[str, Any]]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {"schema_version": "1.0.0", "stages": stages}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return path


def journal_events(work_dir: Path) -> list[dict[str, Any]]:
    return WorkflowJournal(work_dir).records()


def stage_events(work_dir: Path, stage: str, event: str) -> list[dict[str, Any]]:
    return [
        record
        for record in journal_events(work_dir)
        if record.get("stage") == stage and record.get("event") == event
    ]


def failed_envelopes(work_dir: Path, stage: str) -> list[dict[str, Any]]:
    return [record["failure"] for record in stage_events(work_dir, stage, "failed")]


# ---------------------------------------------------------------------------
# F1 (BLOCKER): statistical-gate failures are never retried into a pass
# ---------------------------------------------------------------------------


def test_gate_exit_code_three_stops_at_one_attempt(tmp_path: Path):
    """A stage exiting with the gate exit code must not be retried, even when
    `retries` is declared: attempts stop at 1 and the journal records a
    non-retryable `statistical_gate` envelope."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "gate",
                "command": [
                    sys.executable,
                    "-c",
                    f"import sys; sys.exit({GATE_FAILURE_EXIT_CODE})",
                ],
                "retries": 2,
            }
        ],
    )
    result = run_workflow(workflow, tmp_path / "work")
    assert result["ok"] is False
    assert result["failure_code"] == "statistical_gate"
    assert result["stages"]["gate"]["attempts"] == 1
    envelopes = failed_envelopes(tmp_path / "work", "gate")
    assert len(envelopes) == 1
    assert envelopes[0]["code"] == "statistical_gate"
    assert envelopes[0]["retryable"] is False
    assert envelopes[0]["context"]["attempt"] == 1
    assert envelopes[0]["context"]["log"]


def test_gate_true_stage_failure_is_statistical_gate(tmp_path: Path):
    """A `gate: true` stage failing with a generic nonzero exit is classified
    `statistical_gate` and never retried."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "gate",
                "command": [sys.executable, "-c", "import sys; sys.exit(1)"],
                "gate": True,
            }
        ],
    )
    result = run_workflow(workflow, tmp_path / "work")
    assert result["failure_code"] == "statistical_gate"
    assert result["stages"]["gate"]["attempts"] == 1
    assert failed_envelopes(tmp_path / "work", "gate")[0]["code"] == "statistical_gate"


def test_gate_stage_declares_forbidden_retry_escape_hatches(tmp_path: Path):
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [{"name": "gate", "command": ["true"], "gate": True, "retries": 1}],
    )
    with pytest.raises(ValueError, match="must not declare retries"):
        load_workflow(workflow)
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "gate",
                "command": ["true"],
                "gate": True,
                "continue_on_error": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="may not set continue_on_error"):
        load_workflow(workflow)


def test_cli_exits_three_for_gate_failure(tmp_path: Path):
    """The documented CLI exit-code convention: 3 on statistical-gate failure
    (distinct from 1 for a generic stage failure)."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "gate",
                "command": [sys.executable, "-c", f"import sys; sys.exit({GATE_FAILURE_EXIT_CODE})"],
            }
        ],
    )
    rc = cli_main(
        ["workflow", "run", "--workflow", str(workflow), "--work-dir", str(tmp_path / "wd")]
    )
    assert rc == GATE_FAILURE_EXIT_CODE
    generic = write_workflow(
        tmp_path / "generic.yaml",
        [{"name": "fail", "command": [sys.executable, "-c", "import sys; sys.exit(1)"]}],
    )
    rc = cli_main(
        ["workflow", "run", "--workflow", str(generic), "--work-dir", str(tmp_path / "wd2")]
    )
    assert rc == 1


def test_error_taxonomy_wired_into_retry_decision():
    """classify_exception is live engine code and the retry table is explicit:
    timeouts are transient/retryable, spawn errors are configuration and not
    retryable, statistical gates and cancellation are never retryable."""
    timeout = classify_exception(
        subprocess.TimeoutExpired(cmd=["x"], timeout=1), stage="s"
    )
    assert timeout.code is ErrorCode.TRANSIENT_PROVIDER
    assert timeout.retryable is True
    missing = classify_exception(FileNotFoundError("nope"), stage="s")
    assert missing.code is ErrorCode.CONFIGURATION
    assert missing.retryable is False
    assert classify_exit_code(GATE_FAILURE_EXIT_CODE) is ErrorCode.STATISTICAL_GATE
    assert classify_exit_code(1) is None
    assert retry_decision(ErrorCode.STATISTICAL_GATE, attempts_remaining=True) is False
    assert retry_decision(ErrorCode.USER_CANCELLED, attempts_remaining=True) is False
    assert retry_decision(ErrorCode.CONFIGURATION, attempts_remaining=True) is False
    assert retry_decision(ErrorCode.TRANSIENT_PROVIDER, attempts_remaining=True) is True
    assert retry_decision(ErrorCode.SOFTWARE_DEFECT, attempts_remaining=True) is True
    assert retry_decision(ErrorCode.SOFTWARE_DEFECT, attempts_remaining=False) is False


def test_sigint_journals_user_cancelled(tmp_path: Path):
    """F1/M4: `user_cancelled` is real, not dead taxonomy. SIGINT during a
    stage journals a non-retryable user_cancelled envelope, cleans the lock,
    and the CLI exits 130 with an envelope instead of a traceback."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [{"name": "slow", "command": [sys.executable, "-c", "import time; time.sleep(60)"]}],
    )
    work_dir = tmp_path / "work"
    env = {**os.environ, "PYTHONPATH": str(SRC_ROOT)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "traceguard.cli",
            "workflow",
            "run",
            "--workflow",
            str(workflow),
            "--work-dir",
            str(work_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        journal_path = work_dir / "workflow_events.jsonl"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if journal_path.exists() and '"event": "started"' in journal_path.read_text(
                encoding="utf-8"
            ):
                break
            assert process.poll() is None, "orchestrator exited before the stage started"
            time.sleep(0.05)
        else:
            pytest.fail("stage never started within 20s")
        process.send_signal(signal.SIGINT)
        stdout, _stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:  # pragma: no cover - defensive cleanup
            process.kill()
            process.communicate()
    assert process.returncode == 130
    assert "Traceback" not in stdout
    payload = json.loads(stdout)
    assert payload["code"] == "user_cancelled"
    envelopes = failed_envelopes(work_dir, "slow")
    assert len(envelopes) == 1
    assert envelopes[0]["code"] == "user_cancelled"
    assert envelopes[0]["retryable"] is False
    assert envelopes[0]["exception_type"] == "KeyboardInterrupt"
    assert not (work_dir / "workflow.lock").exists()
    assert (work_dir / "workflow_status.json").exists()


# ---------------------------------------------------------------------------
# F2: stale locks are recovered; live locks are refused
# ---------------------------------------------------------------------------


def test_stale_lock_with_dead_pid_is_taken_over_and_journaled(tmp_path: Path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)
    (work_dir / "workflow.lock").write_text(f"pid={dead.pid}\n", encoding="utf-8")
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('ok')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    result = run_workflow(workflow, work_dir)
    assert result["ok"] is True
    # The takeover is journaled, never silent.
    recovered = [r for r in journal_events(work_dir) if r.get("event") == "lock_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["stale_pid"] == dead.pid


def test_live_lock_holder_is_refused(tmp_path: Path):
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        work_dir = tmp_path / "work"
        work_dir.mkdir(parents=True)
        (work_dir / "workflow.lock").write_text(f"pid={holder.pid}\n", encoding="utf-8")
        workflow = write_workflow(
            tmp_path / "workflow.yaml", [{"name": "noop", "command": ["true"]}]
        )
        with pytest.raises(LockHeldError, match=f"live orchestrator pid={holder.pid}"):
            run_workflow(workflow, work_dir)
        # A refused run must not delete the live holder's lock.
        assert (work_dir / "workflow.lock").exists()
    finally:
        holder.terminate()
        holder.wait()


# ---------------------------------------------------------------------------
# F3: no unjournaled crashes — every failure gets an envelope and a status write
# ---------------------------------------------------------------------------


def test_missing_argv0_is_journaled_cleanly(tmp_path: Path):
    """E22: a command with a nonexistent argv[0] used to crash the orchestrator
    with a raw FileNotFoundError and a dangling `started` event. Now it is a
    journaled, classified, non-retryable failure and the run returns cleanly."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [{"name": "typo", "command": ["traceguardd-does-not-exist", "--oops"], "retries": 2}],
    )
    work_dir = tmp_path / "work"
    result = run_workflow(workflow, work_dir)  # must not raise
    assert result["ok"] is False
    assert result["failure_code"] == "configuration"
    assert result["stages"]["typo"]["attempts"] == 1  # configuration: never retried
    envelopes = failed_envelopes(work_dir, "typo")
    assert len(envelopes) == 1
    assert envelopes[0]["exception_type"] == "FileNotFoundError"
    assert envelopes[0]["retryable"] is False
    log_path = work_dir / envelopes[0]["context"]["log"]
    assert log_path.exists()  # no dangling log references (F4)
    status = json.loads((work_dir / "workflow_status.json").read_text(encoding="utf-8"))
    assert status["ok"] is False
    assert status["failure_code"] == "configuration"


def test_invalid_utf8_output_is_journaled_failure(tmp_path: Path):
    """E10: a stage emitting invalid UTF-8 used to crash the orchestrator
    inside subprocess text decoding. Now the strict decode failure is a
    journaled configuration failure; the raw bytes stay in the attempt log."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "binary",
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'ok\\xff\\xfebad')",
                ],
                "retries": 1,
            }
        ],
    )
    work_dir = tmp_path / "work"
    result = run_workflow(workflow, work_dir)  # must not raise
    assert result["ok"] is False
    assert result["stages"]["binary"]["attempts"] == 1
    envelopes = failed_envelopes(work_dir, "binary")
    assert len(envelopes) == 1
    assert envelopes[0]["exception_type"] == "UnicodeDecodeError"
    assert envelopes[0]["code"] == "configuration"
    assert envelopes[0]["retryable"] is False
    log_path = work_dir / envelopes[0]["context"]["log"]
    raw = log_path.read_bytes()
    assert b"ok\xff\xfebad" in raw  # byte-exact evidence, never "repaired"


def test_continue_on_error_dependent_is_skipped_with_reason(tmp_path: Path):
    """E16/M7: a dependent of a continue_on_error failure used to crash with
    an unjournaled RuntimeError. Now the dependent is journaled as `skipped`
    with the reason and the run returns a status."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "broken",
                "command": [sys.executable, "-c", "import sys; sys.exit(1)"],
                "continue_on_error": True,
            },
            {"name": "dependent", "needs": ["broken"], "command": ["true"]},
        ],
    )
    work_dir = tmp_path / "work"
    result = run_workflow(workflow, work_dir)  # must not raise
    assert result["ok"] is False
    assert result["plan"][1]["action"] == "skipped_dependencies_incomplete"
    assert result["plan"][1]["missing"] == ["broken"]
    skipped = stage_events(work_dir, "dependent", "skipped")
    assert len(skipped) == 1
    assert "broken" in skipped[0]["reason"]
    assert (work_dir / "workflow_status.json").exists()


# ---------------------------------------------------------------------------
# F4: attempt logs always exist, capture partial output, never collide
# ---------------------------------------------------------------------------


def test_timeout_writes_partial_output_to_attempt_log(tmp_path: Path):
    """E7/E19: a timed-out stage used to journal a `failed` event pointing at
    a log file that was never written, destroying the partial output. Now the
    log exists and holds the byte-exact partial stdout."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "slowpoke",
                "command": [
                    sys.executable,
                    "-c",
                    "print('PRE-TIMEOUT EVIDENCE', flush=True); import time; time.sleep(30)",
                ],
                "timeout_seconds": 2,
                "retries": 0,
            }
        ],
    )
    work_dir = tmp_path / "work"
    result = run_workflow(workflow, work_dir)
    assert result["ok"] is False
    assert result["failure"]["code"] == "transient_provider"
    envelopes = failed_envelopes(work_dir, "slowpoke")
    assert len(envelopes) == 1
    log_path = work_dir / envelopes[0]["context"]["log"]
    assert log_path.exists()
    assert b"PRE-TIMEOUT EVIDENCE" in log_path.read_bytes()


def test_resume_never_overwrites_prior_attempt_logs(tmp_path: Path):
    """E9: attempt logs used to be named only by stage+attempt, so a resume
    overwrote the previous run's forensic log. Run nonces make every attempt
    log globally unique across runs."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "failer",
                "command": [
                    sys.executable,
                    "-c",
                    "print('EVIDENCE-FROM-INVOCATION', flush=True); import sys; sys.exit(1)",
                ],
                "retries": 0,
            }
        ],
    )
    work_dir = tmp_path / "work"
    first = run_workflow(workflow, work_dir)
    first_logs = set((work_dir / "logs").glob("*.log"))
    assert len(first_logs) == 1
    first_bytes = {p: p.read_bytes() for p in first_logs}
    assert b"EVIDENCE-FROM-INVOCATION" in next(iter(first_bytes.values()))
    second = run_workflow(workflow, work_dir)
    assert first["ok"] is False and second["ok"] is False
    # The old file is intact and the new run used a different path (set
    # comparison: the nonce is random, so lexicographic order is meaningless).
    second_logs = set((work_dir / "logs").glob("*.log"))
    assert first_logs < second_logs
    assert len(second_logs - first_logs) == 1
    for path, payload in first_bytes.items():
        assert path.read_bytes() == payload
    # Both journal `failed` events and their log paths are distinct.
    log_refs = [e["context"]["log"] for e in failed_envelopes(work_dir, "failer")]
    assert len(set(log_refs)) == 2


# ---------------------------------------------------------------------------
# F5: edited stage commands are never silently skipped
# ---------------------------------------------------------------------------


def test_edited_workflow_resume_is_refused_until_rebaselined(tmp_path: Path):
    """E11: editing a stage command used to resume with `skip_verified`,
    silently reusing outputs produced by the old command. Now the baseline
    mismatch refuses the run; an explicit rebaseline re-runs the stage."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('OLD-LOGIC')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    write_workflow(
        workflow,
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('NEW-LOGIC')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    with pytest.raises(WorkflowBaselineError, match="changed since the journaled run"):
        run_workflow(workflow, work_dir)
    # The stale output is still the old run's: nothing silently re-ran.
    assert (work_dir / "result.txt").read_text(encoding="utf-8") == "OLD-LOGIC"
    # Explicit re-baseline is journaled and re-runs the edited stage.
    result = run_workflow(workflow, work_dir, rebaseline=True)
    assert result["plan"][0]["stage"] == "make"
    assert result["plan"][0]["action"] == "run"
    assert (work_dir / "result.txt").read_text(encoding="utf-8") == "NEW-LOGIC"
    events = journal_events(work_dir)
    assert any(r.get("event") == "rebaselined" for r in events)


def test_cli_refusal_envelope_mentions_recovery_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [{"name": "make", "command": ["true"], "outputs": ["result.txt"]}],
    )
    work_dir = tmp_path / "work"
    run_workflow(workflow, work_dir)
    write_workflow(
        workflow, [{"name": "make", "command": ["false"], "outputs": ["result.txt"]}]
    )
    rc = cli_main(
        ["workflow", "resume", "--workflow", str(workflow), "--work-dir", str(work_dir)]
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "configuration"
    assert "rebaseline" in payload["message"]
    assert "work-dir" in payload["message"]


# ---------------------------------------------------------------------------
# F6: side-effect stages (outputs: []) re-run on resume
# ---------------------------------------------------------------------------


def test_outputless_stage_reruns_on_resume_when_side_effect_lost(tmp_path: Path):
    """E20: an outputs:[] stage whose real side effect was deleted used to be
    skipped on resume because a success record alone was vacuous truth."""
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('side-effect.txt').write_text('done')",
    ]
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [{"name": "monitoring_like", "command": command, "outputs": []}],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    (work_dir / "side-effect.txt").unlink()
    result = run_workflow(workflow, work_dir)
    assert result["plan"][0]["stage"] == "monitoring_like"
    assert result["plan"][0]["action"] == "run"
    assert (work_dir / "side-effect.txt").exists()


def test_idempotent_outputless_stage_skips_on_resume(tmp_path: Path):
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "pure_check",
                "command": ["true"],
                "outputs": [],
                "idempotent": True,
            }
        ],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    result = run_workflow(workflow, work_dir)
    assert result["plan"] == [{"stage": "pure_check", "action": "skip_verified"}]


# ---------------------------------------------------------------------------
# F7: journal tail truncation is detected
# ---------------------------------------------------------------------------


def test_journal_tail_truncation_is_detected(tmp_path: Path):
    """E1c-1: deleting the last journal line used to leave a perfectly valid
    chain that resume accepted silently. The durable anchor now fails loudly."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('ok')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    journal_path = work_dir / "workflow_events.jsonl"
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3
    journal_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(JournalTamperError, match="truncation"):
        run_workflow(workflow, work_dir)
    with pytest.raises(JournalTamperError, match="truncation"):
        workflow_status(workflow, work_dir)


def test_normal_journal_passes_anchor_verification(tmp_path: Path):
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('ok')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    records = journal_events(work_dir)
    state = json.loads((work_dir / "journal_state.json").read_text(encoding="utf-8"))
    assert state["journal_events"] == len(records)
    assert state["last_event_sha256"] == records[-1]["event_sha256"]
    assert records[0]["event"] == "genesis"
    assert workflow_status(workflow, work_dir)["journal_events"] == len(records)


def test_status_file_referencing_beyond_journal_end_fails(tmp_path: Path):
    """F7 belt-and-braces: even if an attacker removes the tail AND the anchor
    state file, the last written status file still references more journal
    events than remain — verification must fail loudly."""
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('ok')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    journal_path = work_dir / "workflow_events.jsonl"
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    journal_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    (work_dir / "journal_state.json").unlink()
    with pytest.raises(JournalTamperError, match="references"):
        run_workflow(workflow, work_dir)


# ---------------------------------------------------------------------------
# F8: timeout kills the whole process group (grandchildren included)
# ---------------------------------------------------------------------------


def test_timeout_kills_entire_process_group(tmp_path: Path):
    """E7-t2: a stage that fans out a grandchild used to leave it orphaned
    after a timeout (VRAM-hogging GPU worker class). The stage runs in its
    own session and the whole group is killed."""
    command = [
        sys.executable,
        "-c",
        "import subprocess, time\n"
        "child = subprocess.Popen(['sleep', '30'])\n"
        "open('grandchild.pid', 'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
    ]
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "fanout",
                "command": command,
                "timeout_seconds": 3,
                "retries": 0,
            }
        ],
    )
    work_dir = tmp_path / "work"
    result = run_workflow(workflow, work_dir)
    assert result["ok"] is False
    grandchild_pid = int((work_dir / "grandchild.pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)  # the grandchild must be dead


# ---------------------------------------------------------------------------
# F9: operator-facing failures are structured envelopes, never raw tracebacks
# ---------------------------------------------------------------------------


def test_tampered_journal_cli_emits_envelope_not_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    workflow = write_workflow(
        tmp_path / "workflow.yaml",
        [
            {
                "name": "make",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('ok')",
                ],
                "outputs": ["result.txt"],
            }
        ],
    )
    work_dir = tmp_path / "work"
    assert run_workflow(workflow, work_dir)["ok"] is True
    journal_path = work_dir / "workflow_events.jsonl"
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["stage"] = "tampered"
    lines[1] = json.dumps(record, sort_keys=True)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc = cli_main(
        ["workflow", "resume", "--workflow", str(workflow), "--work-dir", str(work_dir)]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    payload = json.loads(captured.out)
    assert payload["code"] == "input_integrity"
    assert "digest mismatch" in payload["message"]
    # The traceback is preserved as evidence in the work-dir log, byte-level.
    error_log = work_dir / "logs" / "workflow_cli_errors.log"
    assert b"Traceback" in error_log.read_bytes()


def test_lock_conflict_cli_emits_envelope_not_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        work_dir = tmp_path / "work"
        work_dir.mkdir(parents=True)
        (work_dir / "workflow.lock").write_text(f"pid={holder.pid}\n", encoding="utf-8")
        workflow = write_workflow(
            tmp_path / "workflow.yaml", [{"name": "noop", "command": ["true"]}]
        )
        rc = cli_main(
            ["workflow", "run", "--workflow", str(workflow), "--work-dir", str(work_dir)]
        )
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["code"] == "configuration"
        assert f"pid={holder.pid}" in payload["message"]
    finally:
        holder.terminate()
        holder.wait()
