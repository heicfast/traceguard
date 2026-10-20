"""Evidence inventory and verification for independently auditable runs.

Security-review hardening (docs/reviews/v6_core_security_review.md F2/F6/
F8/F10/F11/F15): symlink refusal, reverse enumeration (unindexed files are
problems, not blind spots), canonical ordering, metadata bound to the run
manifest, NDJSON media types, path-based exclusions so decoy files cannot
escape coverage, and transient operational locks ignored as evidence.
"""

from __future__ import annotations

import hashlib
import json
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

# Operational state, never evidence: a workflow lock present at run time must
# neither be indexed nor required by the ledger (review M6/F2 allow-list).
TRANSIENT_SUFFIXES = (".lock",)


def _is_transient(relative: str) -> bool:
    return relative.endswith(TRANSIENT_SUFFIXES)


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


def _media_type(path: Path) -> str:
    # NDJSON shards are a first-class evidence media type (review F15), not a
    # generic octet-stream.
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".sha256":
        return "text/x-checksum-sha256"
    if path.suffix == ".md":
        return "text/markdown"
    if path.suffix == ".log":
        return "text/plain"
    if path.suffix == ".png":
        return "image/png"
    if path.suffix == ".svg":
        return "image/svg+xml"
    if path.suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _scan_files(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Enumerate indexable files; refuse symlinks loudly (never follow)."""
    files: list[dict[str, Any]] = []
    symlink_problems: list[str] = []
    for path in sorted(item for item in root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlink_problems.append(f"symlink in run directory: {relative}")
            continue
        if not path.is_file():
            continue
        if relative in EXCLUDED_INDEX_PATHS or _is_transient(relative):
            continue
        files.append(
            {
                "path": relative,
                "role": _role(relative),
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
                "media_type": _media_type(path),
            }
        )
    return files, symlink_problems


def build_evidence_index(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest missing: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files, symlink_problems = _scan_files(root)
    if symlink_problems:
        # Fail closed: an in-tree symlink is an evidence red flag; building an
        # index that skips it would silently drop it from coverage.
        raise ValueError("; ".join(symlink_problems))
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

    # Canonical ordering: the files array must be sorted by path (review F10 —
    # a reordered array with recomputed internal digests must not pass).
    paths = [item["path"] for item in files]
    if paths != sorted(paths):
        problems.append("evidence index files not in canonical sorted order")

    # Metadata bound to the run manifest (review F11 — a forged index run_id
    # with a recomputed self-digest must not pass).
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("run_id", "config_sha256", "experiment_stage"):
            expected = run_manifest.get(
                "experiment_stage" if field == "experiment_stage" else field,
                run_manifest.get("stage") if field == "experiment_stage" else None,
            )
            if payload.get(field) != expected:
                problems.append(f"index metadata not bound to run manifest: {field}")

    indexed = {item["path"] for item in files}
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

    # Reverse enumeration (review F2): every real file must be indexed. An
    # added file anywhere in the tree is a problem BY NAME.
    if root.exists():
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                problems.append(f"symlink in run directory: {relative}")
                continue
            if not path.is_file():
                continue
            if relative in EXCLUDED_INDEX_PATHS or _is_transient(relative):
                continue
            if relative not in indexed:
                problems.append(f"unindexed file present: {relative}")

    return {"ok": not problems, "files_checked": len(files), "problems": problems[:100]}


__all__ = ["build_evidence_index", "verify_evidence_index"]
