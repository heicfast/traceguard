"""Resumable, hash-chained workflow execution for TRACEGUARD stages.

Workflow files contain argv arrays, never shell fragments. Each transition is
written to an append-only event journal whose records hash the previous event.
Stages are skipped only when a successful event exists *and* every declared
output still exists with the recorded digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .errors import ErrorCode, FailureEnvelope
from .ids import sha256_json
from .traces.storage import atomic_write_json, atomic_write_text


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    needs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    retries: int = 0
    timeout_seconds: int = 21_600
    continue_on_error: bool = False


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
        )
        for raw in payload["stages"]
    ]


class WorkflowJournal:
    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / "workflow_events.jsonl"

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        previous = "0" * 64
        for index, record in enumerate(records):
            event_hash = record.pop("event_sha256", None)
            if record.get("previous_event_sha256") != previous:
                raise ValueError(f"workflow journal chain broken at event {index}")
            expected = sha256_json(record)
            record["event_sha256"] = event_hash
            if event_hash != expected:
                raise ValueError(f"workflow journal digest mismatch at event {index}")
            previous = event_hash
        return records

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


def run_workflow(
    workflow_path: str | Path,
    work_dir: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = load_workflow(workflow_path)
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    journal = WorkflowJournal(root)
    workflow_digest = sha256_json(payload)
    lock_path = root / "workflow.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"workflow lock already exists: {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        completed: set[str] = set()
        plan: list[dict[str, Any]] = []
        for stage in _stages(payload):
            records = journal.records()
            existing = _successful_output_record(records, stage)
            if _outputs_valid(root, stage, existing):
                completed.add(stage.name)
                plan.append({"stage": stage.name, "action": "skip_verified"})
                continue
            missing_dependencies = set(stage.needs) - completed
            if missing_dependencies:
                raise RuntimeError(
                    f"stage {stage.name} dependencies incomplete: {sorted(missing_dependencies)}"
                )
            plan.append({"stage": stage.name, "action": "run", "command": list(stage.command)})
            if dry_run:
                completed.add(stage.name)
                continue
            succeeded = False
            for attempt in range(1, stage.retries + 2):
                log_path = root / "logs" / f"{stage.name}.attempt-{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
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
                try:
                    process = subprocess.run(
                        list(stage.command),
                        cwd=root,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=stage.timeout_seconds,
                        check=False,
                    )
                    atomic_write_text(log_path, process.stdout)
                    if process.returncode != 0:
                        raise RuntimeError(f"command exited with status {process.returncode}")
                    missing = [
                        relative for relative in stage.outputs if not (root / relative).is_file()
                    ]
                    if missing:
                        raise RuntimeError(f"declared outputs missing: {missing}")
                    digests = {
                        relative: _file_digest(root / relative) for relative in stage.outputs
                    }
                    journal.append(
                        {
                            "event": "succeeded",
                            "stage": stage.name,
                            "attempt": attempt,
                            "duration_seconds": time.monotonic() - started,
                            "output_sha256": digests,
                            "log": str(log_path.relative_to(root)),
                        }
                    )
                    succeeded = True
                    completed.add(stage.name)
                    break
                except (subprocess.TimeoutExpired, RuntimeError) as exc:
                    failure = FailureEnvelope(
                        code=(
                            ErrorCode.TRANSIENT_PROVIDER
                            if isinstance(exc, subprocess.TimeoutExpired)
                            else ErrorCode.SOFTWARE_DEFECT
                        ),
                        message=str(exc),
                        stage=stage.name,
                        exception_type=type(exc).__name__,
                        retryable=attempt <= stage.retries,
                        context={"attempt": attempt, "log": str(log_path.relative_to(root))},
                    )
                    journal.append(
                        {
                            "event": "failed",
                            "stage": stage.name,
                            "attempt": attempt,
                            "failure": failure.to_payload(),
                        }
                    )
            if not succeeded and not stage.continue_on_error:
                break
        status_result = workflow_status(workflow_path, root)
        status_result.update(
            {
                "ok": all(value["complete"] for value in status_result["stages"].values())
                if not dry_run
                else True,
                "dry_run": dry_run,
                "plan": plan,
            }
        )
        atomic_write_json(root / "workflow_status.json", status_result)
        return status_result
    finally:
        if lock_path.exists():
            lock_path.unlink()


__all__ = ["load_workflow", "workflow_status", "run_workflow", "WorkflowJournal"]
