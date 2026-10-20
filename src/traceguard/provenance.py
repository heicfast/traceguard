"""Evidence inventory and verification for independently auditable runs."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ids import sha256_json
from .traces.schema import validate_evidence_index
from .traces.storage import atomic_write_json

EXCLUDED_INDEX_PATHS = {
    "artifacts/checksums.sha256",
    "artifacts/artifact_manifest.json",
    "artifacts/evidence_index.json",
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _role(relative: str) -> str:
    top = relative.split("/", 1)[0]
    return {
        "episodes": "monitor_visible_episode",
        "privileged": "privileged_ground_truth_and_raw_inference",
        "monitor_scores": "monitor_measurement_and_raw_inference",
        "transactions": "write_ahead_transaction",
        "commits": "transaction_commit_marker",
        "reports": "derived_analysis",
        "figures": "derived_visualization",
        "tables": "derived_table",
        "logs": "execution_log",
        "artifacts": "integrity_metadata",
    }.get(top, "run_metadata")


def build_evidence_index(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest missing: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED_INDEX_PATHS:
            continue
        files.append(
            {
                "path": relative,
                "role": _role(relative),
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
                "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            }
        )
    role_counts: dict[str, int] = {}
    for item in files:
        role_counts[item["role"]] = role_counts.get(item["role"], 0) + 1
    content_digest = sha256_json(files)
    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_manifest.get("run_id"),
        "experiment_stage": run_manifest.get("experiment_stage", run_manifest.get("stage")),
        "config_sha256": run_manifest.get("config_sha256"),
        "code_revision": run_manifest.get("code_revision"),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "role_counts": role_counts,
        "content_index_sha256": content_digest,
        "files": files,
        "derivation_contract": {
            "ground_truth": ["privileged/", "transactions/", "commits/"],
            "monitor_inputs": ["episodes/"],
            "measurements": ["monitor_scores/"],
            "derived_outputs": ["reports/", "figures/", "tables/"],
        },
    }
    payload["index_sha256"] = sha256_json(payload)
    validate_evidence_index(payload)
    target = root / "artifacts" / "evidence_index.json"
    atomic_write_json(target, payload)
    return payload


def verify_evidence_index(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    path = root / "artifacts" / "evidence_index.json"
    if not path.exists():
        return {"ok": False, "problems": ["evidence index missing"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        validate_evidence_index(payload)
    except Exception as exc:
        return {"ok": False, "problems": [f"evidence index schema violation: {exc}"]}
    claimed = payload.pop("index_sha256", None)
    problems: list[str] = []
    if claimed != sha256_json(payload):
        problems.append("evidence index self-digest mismatch")
    files = payload.get("files", [])
    if payload.get("content_index_sha256") != sha256_json(files):
        problems.append("evidence content-list digest mismatch")
    for item in files:
        target = (root / item["path"]).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            problems.append(f"path escapes run root: {item['path']}")
            continue
        if not target.is_file():
            problems.append(f"indexed file missing: {item['path']}")
        elif _digest(target) != item["sha256"]:
            problems.append(f"indexed digest mismatch: {item['path']}")
    return {"ok": not problems, "files_checked": len(files), "problems": problems[:100]}


__all__ = ["build_evidence_index", "verify_evidence_index"]
