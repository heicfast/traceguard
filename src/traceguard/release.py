"""Build a self-verifying research archive from a completed run directory."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .ids import sha256_json, sha256_text
from .traces.schema import validate_monitor_score, validate_privileged
from .traces.storage import RunStore, atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: record is not an object")
        records.append(payload)
    return records


def _audit_hash_errors(value: Any, context: str) -> tuple[list[str], int]:
    """Recursively validate any embedded ModelTurn audit payloads."""
    errors: list[str] = []
    checked = 0
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_errors, child_checked = _audit_hash_errors(item, f"{context}[{index}]")
            errors.extend(child_errors)
            checked += child_checked
    elif isinstance(value, dict):
        if "raw_output" in value and "raw_output_utf8_sha256" in value:
            checked += 1
            checks = {
                "raw_output_utf8_sha256": sha256_text(value.get("raw_output", "")),
                "raw_output_with_special_tokens_utf8_sha256": sha256_text(
                    value.get("raw_output_with_special_tokens", "")),
                "rendered_prompt_utf8_sha256": sha256_text(value.get("rendered_prompt", "")),
                "input_messages_sha256": sha256_json(value.get("input_messages", [])),
                "prompt_token_ids_sha256": sha256_json(value.get("prompt_token_ids", [])),
                "generated_token_ids_sha256": sha256_json(
                    value.get("generated_token_ids", [])),
            }
            for key, expected in checks.items():
                if value.get(key) != expected:
                    errors.append(f"{context}: audit hash mismatch for {key}")
        for key, item in value.items():
            child_errors, child_checked = _audit_hash_errors(item, f"{context}.{key}")
            errors.extend(child_errors)
            checked += child_checked
    return errors, checked


def build_release(
    run_dir: str | Path,
    *,
    archive_path: str | Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate raw evidence coverage, emit a manifest, and optionally archive it.

    Research-stage releases fail closed if episode, privileged, monitor-score,
    or analysis artifacts are missing. ``allow_incomplete`` exists only for
    operational checkpoints and is recorded prominently in the manifest.
    """
    store = RunStore(run_dir)
    root = store.run_dir.resolve()
    manifest = store.read_run_manifest()
    episodes = store.load_all_episodes(validate=True)
    privileged = store.load_all_privileged()
    completed = store.completed_episode_ids()
    errors: list[str] = []

    episode_ids = {record["episode_id"] for record in episodes}
    if len(episode_ids) != len(episodes):
        errors.append("duplicate episode_id in episode shards")
    if episode_ids != set(privileged):
        errors.append("episode and privileged record id sets differ")
    if episode_ids != completed:
        errors.append("episode records and completion ledger differ")
    for record in privileged.values():
        validate_privileged(record)

    score_files = sorted(store.scores_dir.glob("*.jsonl"))
    score_count = 0
    raw_score_count = 0
    monitor_audit_turns_checked = 0
    score_groups: dict[str, int] = {}
    for path in score_files:
        records = _jsonl_records(path)
        score_groups[path.name] = len(records)
        score_count += len(records)
        for record in records:
            validate_monitor_score(record)
            required_raw = ("raw_response", "rendered_prompt", "inference_audit",
                            "view_hash", "prompt_hash")
            if all(key in record and record[key] is not None for key in required_raw):
                raw_score_count += 1
            else:
                errors.append(f"score lacks raw audit fields: {record.get('score_id', '?')}")
            if sha256_text(record.get("rendered_prompt", "")) != record.get(
                    "rendered_prompt_hash"):
                errors.append(f"rendered prompt hash mismatch: {record.get('score_id', '?')}")
            audit_errors, audit_checked = _audit_hash_errors(
                record.get("inference_audit"), record.get("score_id", "unknown-score"))
            errors.extend(audit_errors)
            monitor_audit_turns_checked += audit_checked

    report_paths = sorted((root / "reports").glob("*")) if (root / "reports").exists() else []
    stage = manifest.get("experiment_stage", manifest.get("stage", "unknown"))
    if stage in {"pilot", "confirmatory", "online"} and not allow_incomplete:
        if not score_files:
            errors.append("research release has no monitor score shards")
        if not (root / "reports" / "analysis_report.json").exists():
            errors.append("research release has no analysis_report.json")
    if errors and not allow_incomplete:
        raise ValueError("release validation failed: " + "; ".join(sorted(set(errors))))

    excluded = {
        Path("artifacts/artifact_manifest.json"),
        Path("artifacts/checksums.sha256"),
    }
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative in excluded:
            continue
        files.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })

    release_manifest = {
        "schema_version": "1.0.0",
        "release_complete": not errors,
        "allow_incomplete": allow_incomplete,
        "validation_errors": sorted(set(errors)),
        "run_id": manifest.get("run_id"),
        "experiment_stage": stage,
        "config_sha256": manifest.get("config_sha256"),
        "code_revision": manifest.get("code_revision"),
        "counts": {
            "episodes": len(episodes),
            "privileged_records": len(privileged),
            "completed_ids": len(completed),
            "monitor_scores": score_count,
            "monitor_scores_with_raw_audit": raw_score_count,
            "monitor_inference_turn_hashes_checked": monitor_audit_turns_checked,
            "monitor_score_shards": score_groups,
            "report_files": len([path for path in report_paths if path.is_file()]),
        },
        "files": files,
    }
    manifest_path = root / "artifacts" / "artifact_manifest.json"
    atomic_write_json(manifest_path, release_manifest)
    store.write_checksums()

    archive_result: str | None = None
    archive_sha256: str | None = None
    if archive_path is not None:
        target = Path(archive_path).resolve()
        if target == root or root in target.parents:
            raise ValueError("archive path must be outside the run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=".tmp-release-", suffix=".tar.gz")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
                archive.add(root, arcname=root.name, recursive=True)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        archive_result = str(target)
        archive_sha256 = _sha256(target)

    return {
        "ok": not errors,
        "manifest_path": str(manifest_path),
        "archive_path": archive_result,
        "archive_sha256": archive_sha256,
        "manifest": release_manifest,
    }


__all__ = ["build_release"]
