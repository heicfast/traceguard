"""Resumable, hash-chained workflow execution for TRACEGUARD stages.

Workflow files contain argv arrays, never shell fragments. Each transition is
written to an append-only event journal whose records hash the previous event.
Stages are skipped only when a successful event exists *and* every declared
output still exists with the recorded digest.

Engine contracts fixed by the 2026-10 v0.6.0 developer review (F1-F9):

* **Gate failures are never retried into a pass (F1).** A stage whose command
  exits with the predeclared gate exit code (``errors.GATE_FAILURE_EXIT_CODE``
  = 3), or any failure of a stage declared ``gate: true``, is classified
  ``statistical_gate`` — an absolute non-retryable class. Gate stages must
  declare ``retries: 0`` and may not set ``continue_on_error``. All engine
  exceptions are classified through :func:`traceguard.errors.classify_exception`;
  the retry decision follows the two-layer policy documented in
  ``errors.py`` (class-intrinsic retryability for engine-detected failures,
  operator-declared ``retries`` for stage exit-code verdicts, never for
  ``statistical_gate``/``user_cancelled``).
* **Stale locks are recovered, live locks are refused (F2).** The lock records
  the holder PID. If the holder PID is dead, a new run takes over the lock and
  journals a ``lock_recovered`` event; if it is alive (or the lock is not a
  readable pid file, which is how a just-created lock looks), the new run
  refuses.
* **No unjournaled crashes (F3).** Every stage attempt is wrapped in a broad
  ``except Exception`` (never ``BaseException``): every failure gets a failure
  envelope in the journal (stage, exception type, retryability, attempt, log
  path) and a ``workflow_status.json`` write. A stage whose declared
  dependencies are incomplete is journaled as ``skipped`` with the reason
  instead of crashing unjournaled.
* **Attempt logs always exist and never collide (F4).** Captured output —
  including partial output of a timed-out or interrupted stage — is written
  byte-exactly to ``logs/<stage>.run-<nonce>.attempt-<n>.log``. The run nonce
  is unique per invocation, so a resume never overwrites prior forensic logs.
* **Edited workflows never silently reuse stale outputs (F5).** The journal's
  active baseline digest must match the current workflow file. A mismatch
  refuses the run with a clear error; the operator either starts a new
  work-dir or explicitly re-baselines (``rebaseline=True`` / CLI
  ``--rebaseline``), which journals a ``rebaselined`` event. Success records
  from a previous baseline are never used for skipping.
* **Side-effect-only stages re-run on resume (F6).** A stage with no declared
  outputs has side effects the journal cannot verify, so resume re-runs it
  unless the operator declared it safe to skip with ``idempotent: true`` (or
  combined it with ``continue_on_error``, the legacy operator-acknowledged
  flag). Declare real outputs (or ``idempotent``) for expensive stages.
* **Journal tail truncation is detected (F7).** Every append durably records
  the journal length and last event hash in ``journal_state.json``; every read
  verifies the journal against that anchor (and against
  ``workflow_status.json``). A journal shorter than its anchor fails loudly as
  tampering. The first journal record is a ``genesis`` event carrying the
  workflow SHA-256, and the hash chain is verified from it.
* **Timeouts kill the whole process group (F8).** Stages run with
  ``start_new_session=True``; on timeout (and on operator interrupt) the
  entire group — the direct child and any fan-out grandchildren such as GPU
  workers — receives SIGTERM and then SIGKILL, so nothing is orphaned.
* **Cancellation is journaled (F1/M4).** SIGINT during a stage kills the
  group, journals a non-retryable ``user_cancelled`` failure envelope, writes
  the status file, and re-raises.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .errors import (
    ErrorCode,
    FailureEnvelope,
    classify_exception,
    classify_exit_code,
    retry_decision,
)
from .ids import sha256_json
from .traces.storage import atomic_write_json, atomic_write_text


class JournalTamperError(ValueError):
    """Journal content, chain, or anchor verification failed (input integrity)."""


class LockHeldError(RuntimeError):
    """The workflow lock is held by a live or unidentifiable process."""


class WorkflowBaselineError(RuntimeError):
    """The workflow file no longer matches the journal's active baseline."""


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durable atomic byte-exact write (temp file, fsync, rename, dir fsync)."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        if os.name == "posix":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    needs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    retries: int = 0
    timeout_seconds: int = 21_600
    continue_on_error: bool = False
    gate: bool = False
    idempotent: bool = False


def load_workflow(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ValueError("workflow must be a schema_version 1.0.0 mapping")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("workflow must declare at least one stage")
    names: set[str] = set()
    for raw in raw_stages:
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError("every workflow stage requires a name")
        if raw["name"] in names:
            raise ValueError(f"duplicate workflow stage: {raw['name']}")
        names.add(raw["name"])
        command = raw.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError(f"stage {raw['name']} command must be a nonempty argv list")
        for output in raw.get("outputs", []):
            output_path = Path(output)
            if output_path.is_absolute() or ".." in output_path.parts:
                raise ValueError(
                    f"stage {raw['name']} output must stay inside the workflow root: {output}"
                )
        for flag in ("gate", "idempotent"):
            if flag in raw and not isinstance(raw[flag], bool):
                raise ValueError(f"stage {raw['name']} {flag} must be a boolean")
        if raw.get("gate"):
            if int(raw.get("retries", 0)) != 0:
                raise ValueError(
                    f"gate stage {raw['name']} must not declare retries: "
                    "statistical gate failures are never retried"
                )
            if raw.get("continue_on_error"):
                raise ValueError(f"gate stage {raw['name']} may not set continue_on_error")
    for raw in raw_stages:
        unknown = set(raw.get("needs", [])) - names
        if unknown:
            raise ValueError(f"stage {raw['name']} has unknown dependencies: {sorted(unknown)}")
        if raw["name"] in raw.get("needs", []):
            raise ValueError(f"stage {raw['name']} depends on itself")
    return payload


def _stages(payload: dict[str, Any]) -> list[Stage]:
    return [
        Stage(
            name=raw["name"],
            command=tuple(raw["command"]),
            needs=tuple(raw.get("needs", [])),
            outputs=tuple(raw.get("outputs", [])),
            retries=int(raw.get("retries", 0)),
            timeout_seconds=int(raw.get("timeout_seconds", 21_600)),
            continue_on_error=bool(raw.get("continue_on_error", False)),
            gate=bool(raw.get("gate", False)),
            idempotent=bool(raw.get("idempotent", False)),
        )
        for raw in payload["stages"]
    ]


class WorkflowJournal:
    """Hash-chained, tail-anchored append-only event journal (F7).

    Every append durably records the resulting journal length and last event
    hash in ``journal_state.json``. Every read re-verifies the full chain and
    then the anchors: a journal shorter than its recorded anchor is tail
    truncation (tamper), and a head hash mismatch means the retained prefix
    was rewritten. Crash-consistency note: the journal file is written first
    and the anchor second, so a crash between the two leaves a journal longer
    than its anchor, which is accepted (the next append re-anchors).
    """

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / "workflow_events.jsonl"
        self.state_path = self.work_dir / "journal_state.json"

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            records: list[dict[str, Any]] = []
        else:
            records = []
            for index, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines()
            ):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise JournalTamperError(
                        f"workflow journal line {index} is not valid JSON: {exc}"
                    ) from exc
            previous = "0" * 64
            for index, record in enumerate(records):
                event_hash = record.pop("event_sha256", None)
                if record.get("previous_event_sha256") != previous:
                    raise JournalTamperError(f"workflow journal chain broken at event {index}")
                expected = sha256_json(record)
                record["event_sha256"] = event_hash
                if event_hash != expected:
                    raise JournalTamperError(
                        f"workflow journal digest mismatch at event {index}"
                    )
                previous = event_hash
        self._verify_anchors(records)
        return records

    def _verify_anchors(self, records: list[dict[str, Any]]) -> None:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                anchored_count = int(state["journal_events"])
                anchored_hash = str(state["last_event_sha256"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise JournalTamperError(
                    f"journal anchor state is unreadable: {self.state_path}: {exc}"
                ) from exc
            if len(records) < anchored_count:
                raise JournalTamperError(
                    f"journal tail truncation detected: {len(records)} events present but "
                    f"journal_state.json anchors {anchored_count} events "
                    f"(last anchored hash {anchored_hash[:16]})"
                )
            if anchored_count >= 1 and records[anchored_count - 1].get("event_sha256") != (
                anchored_hash
            ):
                raise JournalTamperError(
                    "journal/state anchor mismatch: the journal head does not match "
                    "the hash anchored in journal_state.json"
                )
        status_path = self.work_dir / "workflow_status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                recorded = int(status.get("journal_events", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                return  # derived output; the journal and anchor checks above are authoritative
            if recorded > len(records):
                raise JournalTamperError(
                    f"workflow_status.json references {recorded} journal events but only "
                    f"{len(records)} are present"
                )

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        records = self.records()
        previous = records[-1]["event_sha256"] if records else "0" * 64
        payload = {
            "schema_version": "1.0.0",
            "sequence": len(records),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "previous_event_sha256": previous,
            **event,
        }
        payload["event_sha256"] = sha256_json(payload)
        lines = [json.dumps(record, sort_keys=True) for record in records]
        lines.append(json.dumps(payload, sort_keys=True))
        atomic_write_text(self.path, "\n".join(lines) + "\n")
        atomic_write_json(
            self.state_path,
            {
                "schema_version": "1.0.0",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "journal_events": len(records) + 1,
                "last_event_sha256": payload["event_sha256"],
            },
        )
        return payload


def _successful_output_record(records: list[dict[str, Any]], stage: Stage) -> dict | None:
    for record in reversed(records):
        if record.get("stage") == stage.name and record.get("event") == "succeeded":
            return record
    return None


def _outputs_valid(root: Path, stage: Stage, record: dict[str, Any] | None) -> bool:
    if record is None:
        return False
    stored = record.get("output_sha256", {})
    for relative in stage.outputs:
        path = root / relative
        if not path.is_file() or stored.get(relative) != _file_digest(path):
            return False
    return True


def _active_baseline_digest(records: list[dict[str, Any]]) -> str | None:
    """The workflow SHA-256 the journal was last written under (F5).

    ``rebaselined`` events transition the baseline; every ``started``/
    ``succeeded``/``genesis`` event records the digest of the run that wrote
    it. Because runs refuse on mismatch, the last observed digest is the
    active baseline.
    """
    digest: str | None = None
    for record in records:
        if record.get("event") == "rebaselined":
            digest = record.get("to_workflow_sha256")
        elif "workflow_sha256" in record:
            digest = record["workflow_sha256"]
    return digest


def _stage_is_skip_verified(
    root: Path, stage: Stage, records: list[dict[str, Any]], workflow_digest: str
) -> bool:
    record = _successful_output_record(records, stage)
    if record is None:
        return False
    if record.get("workflow_sha256") != workflow_digest:
        # Success record predates the current workflow baseline (F5): outputs
        # produced by an older stage command must never be silently reused.
        return False
    if not stage.outputs and not (stage.idempotent or stage.continue_on_error):
        # Side-effect-only stage (F6): with no declared outputs the journal
        # cannot verify the stage's real side effects, so resume re-runs it
        # unless the operator explicitly declared it skippable.
        return False
    return _outputs_valid(root, stage, record)


def _read_lock_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("pid="):
        return None
    digits = text[len("pid=") :]
    return int(digits) if digits.isdigit() else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def _replace_lock(lock_path: Path, pid: int) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(lock_path.parent), prefix=".lock-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={pid}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, lock_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _acquire_lock(lock_path: Path) -> tuple[bool, int | None]:
    """Take the workflow lock (F2). Returns ``(owned, stale_holder_pid)``.

    If an existing lock names a dead PID, the lock is taken over and the dead
    holder's PID is returned for journaling. A live holder — or a lock without
    a readable PID, which is how a just-created lock looks mid-write — is
    refused with :class:`LockHeldError`.

    Residual race (documented, accepted): two resumes that both observe the
    same dead PID could interleave takeover writes. The post-takeover re-read
    makes the last writer the single owner and the loser refuses; a takeover
    starting after another's verification window is theoretically possible on
    a shared host. The journal hash chain makes any colliding append a loud
    chain/anchor error rather than a silent corruption.
    """
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        holder = _read_lock_pid(lock_path)
        if holder is None:
            raise LockHeldError(
                "workflow lock exists but carries no readable pid (a concurrent "
                f"orchestrator may have just created it): {lock_path}; if no "
                "orchestrator is running, remove the lock file and retry"
            ) from None
        if _pid_alive(holder):
            raise LockHeldError(
                f"workflow lock held by live orchestrator pid={holder}: {lock_path}; "
                "stop that process or wait for it to finish"
            ) from None
        _replace_lock(lock_path, os.getpid())
        if _read_lock_pid(lock_path) != os.getpid():
            raise LockHeldError(
                f"lost the stale-lock takeover race on {lock_path}: another "
                "orchestrator took it over first"
            ) from None
        return True, holder
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
    finally:
        os.close(descriptor)
    return True, None


def _terminate_process_group(
    process: subprocess.Popen[Any], grace_seconds: float = 2.0
) -> None:
    """Kill the stage's whole process group: child plus grandchildren (F8).

    Stages run with ``start_new_session=True`` so the group is exactly the
    stage's process tree (GPU worker fan-out included). SIGTERM first for a
    short grace period, then an unconditional SIGKILL to the group — a
    grandchild that ignores SIGTERM must not survive the attempt.
    """
    try:
        group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group_id, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_seconds)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group_id, signal.SIGKILL)
    # A SIGKILLed group is reaped immediately; the bound only guards against
    # an uninterruptible (D-state) direct child hanging the orchestrator.
    with contextlib.suppress(subprocess.TimeoutExpired):  # pragma: no cover - D-state
        process.wait(timeout=10)


def _record_stage_failure(
    journal: WorkflowJournal,
    stage: Stage,
    attempt: int,
    relative_log: str,
    code: ErrorCode,
    message: str,
    exception_type: str | None,
    exit_status: int | None = None,
) -> FailureEnvelope:
    """Build the envelope, apply the retry decision, journal it (F1/F3)."""
    context: dict[str, Any] = {"attempt": attempt, "log": relative_log}
    if exit_status is not None:
        context["exit_status"] = exit_status
    failure = FailureEnvelope(
        code=code,
        message=message,
        stage=stage.name,
        exception_type=exception_type,
        retryable=retry_decision(code, attempts_remaining=attempt <= stage.retries),
        context=context,
    )
    journal.append(
        {
            "event": "failed",
            "stage": stage.name,
            "attempt": attempt,
            "failure": failure.to_payload(),
        }
    )
    return failure


def _execute_attempt(
    stage: Stage,
    root: Path,
    journal: WorkflowJournal,
    workflow_digest: str,
    run_nonce: str,
    attempt: int,
) -> FailureEnvelope | None:
    """Run one stage attempt. Returns ``None`` on success, else the envelope.

    Every failure — nonzero exit, timeout, spawn error, undecodable output,
    missing declared outputs, operator interrupt — is journaled with a
    failure envelope before this function returns or re-raises (F3). The
    captured output (partial output included on timeout/interrupt) is always
    written byte-exactly to the attempt log (F4), and the stage runs in its
    own session so the whole group dies on timeout (F8).
    """
    log_path = root / "logs" / f"{stage.name}.run-{run_nonce}.attempt-{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    relative_log = str(log_path.relative_to(root))
    journal.append(
        {
            "event": "started",
            "stage": stage.name,
            "attempt": attempt,
            "workflow_sha256": workflow_digest,
            "command": list(stage.command),
        }
    )
    started = time.monotonic()
    captured = b""
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(stage.command),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            captured, _ = process.communicate(timeout=stage.timeout_seconds)
            captured = captured or b""
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            try:
                captured, _ = process.communicate(timeout=30)
                captured = captured or b""
            except subprocess.TimeoutExpired as drained:
                captured = drained.stdout or b""
            raise subprocess.TimeoutExpired(
                cmd=list(stage.command), timeout=stage.timeout_seconds
            ) from None
        returncode = process.returncode
        if returncode != 0:
            gate_failure = stage.gate or classify_exit_code(returncode) is not None
            code = ErrorCode.STATISTICAL_GATE if gate_failure else ErrorCode.SOFTWARE_DEFECT
            message = f"stage command exited with status {returncode}"
            if gate_failure:
                message += " (statistical gate failure: never retried)"
            return _record_stage_failure(
                journal, stage, attempt, relative_log, code, message,
                exception_type=None, exit_status=returncode,
            )
        # Strict UTF-8 validation of captured output: malformed output is a
        # journaled stage failure, never silently repaired or replaced (NF-3/7).
        captured.decode("utf-8")
        missing = [
            relative for relative in stage.outputs if not (root / relative).is_file()
        ]
        if missing:
            return _record_stage_failure(
                journal, stage, attempt, relative_log, ErrorCode.SOFTWARE_DEFECT,
                f"declared outputs missing: {missing}", exception_type=None,
            )
        digests = {
            relative: _file_digest(root / relative) for relative in stage.outputs
        }
        journal.append(
            {
                "event": "succeeded",
                "stage": stage.name,
                "attempt": attempt,
                "duration_seconds": time.monotonic() - started,
                "workflow_sha256": workflow_digest,
                "output_sha256": digests,
                "log": relative_log,
            }
        )
        return None
    except KeyboardInterrupt:
        if process is not None:
            _terminate_process_group(process)
        _record_stage_failure(
            journal, stage, attempt, relative_log, ErrorCode.USER_CANCELLED,
            "orchestrator interrupted (SIGINT) before stage completion",
            exception_type="KeyboardInterrupt",
        )
        raise
    except Exception as exc:
        failure = classify_exception(exc, stage=stage.name)
        failure = replace(
            failure,
            retryable=retry_decision(
                failure.code, attempts_remaining=attempt <= stage.retries
            ),
            context={"attempt": attempt, "log": relative_log},
        )
        journal.append(
            {
                "event": "failed",
                "stage": stage.name,
                "attempt": attempt,
                "failure": failure.to_payload(),
            }
        )
        return failure
    finally:
        # F4: the attempt log always exists, even on timeout or spawn failure,
        # and holds the byte-exact captured output (partial output included).
        _atomic_write_bytes(log_path, captured)


def workflow_status(workflow_path: str | Path, work_dir: str | Path) -> dict[str, Any]:
    payload = load_workflow(workflow_path)
    root = Path(work_dir)
    journal = WorkflowJournal(root)
    records = journal.records()
    status: dict[str, Any] = {}
    for stage in _stages(payload):
        success = _successful_output_record(records, stage)
        status[stage.name] = {
            "complete": _outputs_valid(root, stage, success),
            "attempts": sum(
                record.get("stage") == stage.name and record.get("event") == "started"
                for record in records
            ),
            "last_event": next(
                (
                    record.get("event")
                    for record in reversed(records)
                    if record.get("stage") == stage.name
                ),
                None,
            ),
        }
    return {
        "schema_version": "1.0.0",
        "workflow_sha256": sha256_json(payload),
        "stages": status,
        "journal_events": len(records),
    }


def _write_status(
    root: Path,
    workflow_path: str | Path,
    *,
    dry_run: bool,
    plan: list[dict[str, Any]],
    stopping_failure: FailureEnvelope | None,
) -> dict[str, Any]:
    status_result = workflow_status(workflow_path, root)
    status_result.update(
        {
            "ok": (
                all(value["complete"] for value in status_result["stages"].values())
                if not dry_run
                else True
            ),
            "dry_run": dry_run,
            "plan": plan,
            "failure_code": stopping_failure.code.value if stopping_failure else None,
            "failure": stopping_failure.to_payload() if stopping_failure else None,
        }
    )
    atomic_write_json(root / "workflow_status.json", status_result)
    return status_result


def run_workflow(
    workflow_path: str | Path,
    work_dir: str | Path,
    *,
    dry_run: bool = False,
    rebaseline: bool = False,
) -> dict[str, Any]:
    """Run (or resume, or dry-run plan) a workflow against ``work_dir``.

    ``run`` and ``resume`` share semantics. ``rebaseline=True`` explicitly
    re-baselines the journal after a workflow edit (F5); without it, a
    workflow whose SHA-256 no longer matches the journal baseline is refused.
    """
    payload = load_workflow(workflow_path)
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    journal = WorkflowJournal(root)
    workflow_digest = sha256_json(payload)
    lock_path = root / "workflow.lock"
    lock_owned, stale_holder_pid = _acquire_lock(lock_path)
    try:
        records = journal.records()
        baseline = _active_baseline_digest(records)
        if baseline is not None and baseline != workflow_digest:
            if not rebaseline:
                raise WorkflowBaselineError(
                    "workflow file changed since the journaled run "
                    f"(journaled sha256 {baseline}, current {workflow_digest}); "
                    "refusing to run against a stale journal: start a new work-dir, "
                    "or explicitly re-baseline with rebaseline=True (CLI: --rebaseline)"
                )
            if not dry_run:
                journal.append(
                    {
                        "event": "rebaselined",
                        "from_workflow_sha256": baseline,
                        "to_workflow_sha256": workflow_digest,
                    }
                )
        if not dry_run:
            if not records:
                journal.append({"event": "genesis", "workflow_sha256": workflow_digest})
            if stale_holder_pid is not None:
                journal.append({"event": "lock_recovered", "stale_pid": stale_holder_pid})
        run_nonce = uuid.uuid4().hex[:12]
        completed: set[str] = set()
        plan: list[dict[str, Any]] = []
        stopping_failure: FailureEnvelope | None = None
        try:
            for stage in _stages(payload):
                records = journal.records()
                if _stage_is_skip_verified(root, stage, records, workflow_digest):
                    completed.add(stage.name)
                    plan.append({"stage": stage.name, "action": "skip_verified"})
                    continue
                missing_dependencies = set(stage.needs) - completed
                if missing_dependencies:
                    reason = f"dependencies incomplete: {sorted(missing_dependencies)}"
                    if not dry_run:
                        journal.append({"event": "skipped", "stage": stage.name, "reason": reason})
                    plan.append(
                        {
                            "stage": stage.name,
                            "action": "skipped_dependencies_incomplete",
                            "missing": sorted(missing_dependencies),
                        }
                    )
                    continue
                plan.append({"stage": stage.name, "action": "run", "command": list(stage.command)})
                if dry_run:
                    completed.add(stage.name)
                    continue
                stage_failure: FailureEnvelope | None = None
                for attempt in range(1, stage.retries + 2):
                    stage_failure = _execute_attempt(
                        stage, root, journal, workflow_digest, run_nonce, attempt
                    )
                    if stage_failure is None:
                        completed.add(stage.name)
                        break
                    if not stage_failure.retryable:
                        break
                if stage_failure is not None:
                    stopping_failure = stage_failure
                    if not stage.continue_on_error:
                        break
        except KeyboardInterrupt:
            stopping_failure = FailureEnvelope(
                code=ErrorCode.USER_CANCELLED,
                message="orchestrator interrupted (SIGINT)",
                stage="<orchestrator>",
                exception_type="KeyboardInterrupt",
                retryable=False,
            )
            _write_status(
                root, workflow_path, dry_run=dry_run, plan=plan,
                stopping_failure=stopping_failure,
            )
            raise
        return _write_status(
            root, workflow_path, dry_run=dry_run, plan=plan,
            stopping_failure=stopping_failure,
        )
    finally:
        if lock_owned and lock_path.exists():
            lock_path.unlink()


__all__ = [
    "JournalTamperError",
    "LockHeldError",
    "WorkflowBaselineError",
    "load_workflow",
    "workflow_status",
    "run_workflow",
    "WorkflowJournal",
]
