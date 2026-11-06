"""Evidence sync to object storage (S3) with digest verification.

Run directories are the canonical evidence store; this module mirrors them
to S3 for durability and sharing WITHOUT changing the local contract:

* every file is uploaded with its SHA-256 in the object metadata;
* after upload, the object's ETag/content is re-downloaded and digested
  (verify-after-write), so a corrupted transfer is caught immediately;
* the sync writes a ``sync_manifest.json`` mapping every uploaded key to its
  digest and size, anchored to the run's config hash and code revision;
* credentials come exclusively from the environment / standard AWS config
  chain — never from repository files.

Requires ``boto3`` (extras: ``traceguard[aws]``); the module fails closed
with a clear error when it is absent.  Buckets are addressed as
``s3://bucket/prefix``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ids import sha256_json
from .traces.storage import atomic_write_json

CHUNK_SIZE = 1024 * 1024


def _digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            value.update(chunk)
    return value.hexdigest()


def _resolve_s3_path(target: str) -> tuple[str, str]:
    if not target.startswith("s3://"):
        raise ValueError(f"target must be s3://bucket/prefix, got {target!r}")
    rest = target[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError("s3 target has no bucket")
    return bucket, prefix.rstrip("/")


def sync_run_to_s3(run_dir: str | Path, target: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Upload every file of a run directory to S3 with verify-after-write."""
    try:
        import boto3  # noqa: F401
        from boto3.s3.transfer import TransferConfig
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "evidence sync requires boto3: pip install 'traceguard[aws]'"
        ) from exc

    root = Path(run_dir).resolve()
    if not (root / "run_manifest.json").exists():
        raise FileNotFoundError(f"not a run directory (no run_manifest.json): {root}")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    bucket, prefix = _resolve_s3_path(target)

    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    entries: list[dict[str, Any]] = []
    if dry_run:
        for path in files:
            entries.append({
                "key": f"{prefix}/{path.relative_to(root).as_posix()}",
                "bytes": path.stat().st_size,
                "sha256": _digest_file(path),
                "uploaded": False,
            })
        return {"ok": True, "dry_run": True, "bucket": bucket, "files": entries}

    import boto3

    client = boto3.client("s3")
    transfer = TransferConfig(multipart_chunksize=64 * 1024 * 1024)
    problems: list[str] = []
    for path in files:
        key = f"{prefix}/{path.relative_to(root).as_posix()}"
        digest = _digest_file(path)
        client.upload_file(
            str(path), bucket, key,
            ExtraArgs={"Metadata": {"sha256": digest}},
            Config=transfer,
        )
        # verify-after-write: re-download and digest
        downloaded = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if hashlib.sha256(downloaded).hexdigest() != digest:
            problems.append(f"digest mismatch after upload: {key}")
            continue
        entries.append({"key": key, "bytes": path.stat().st_size,
                        "sha256": digest, "uploaded": True})

    payload = {
        "schema_version": "1.0.0",
        "synced_at_utc": datetime.now(UTC).isoformat(),
        "bucket": bucket,
        "prefix": prefix,
        "run_id": manifest.get("run_id"),
        "config_sha256": manifest.get("config_sha256"),
        "code_revision": manifest.get("code_revision"),
        "file_count": len(entries),
        "files": entries,
        "problems": problems,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    atomic_write_json(root / "artifacts" / "sync_manifest.json", payload)
    return {"ok": not problems, "bucket": bucket, "prefix": prefix,
            "files": len(entries), "problems": problems[:20]}


__all__ = ["sync_run_to_s3"]
