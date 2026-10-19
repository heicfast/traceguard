"""Immutable evidence storage: atomic writes, shards, checksums, resumability.

Layout of a run directory (docs/03_KAGGLE_RUNBOOK.md §8):

    <run_dir>/
      run_manifest.json            # frozen config + provenance
      plan_manifest.json           # planned episode set (pre-execution)
      environment.json             # GPU/runtime probe (when present)
      completed_episode_ids.txt    # resumability ledger (append-only)
      episodes/shard-00001.jsonl   # privileged envelopes (schema-validated)
      privileged/shard-00001.jsonl# audit store: raw outputs, hidden text, transitions
      monitor_scores/<monitor>-<condition>-00001.jsonl
      logs/stderr.log
      artifacts/checksums.sha256

Write discipline: every record is written via atomic temp-file -> fsync ->
rename. Resume discipline: scan completed episode ids; skip complete valid
records; never overwrite. Nothing is ever silently dropped or duplicated.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..ids import sha256_json


def atomic_write_text(path: str | Path, text: str) -> None:
    """Durable atomic write: temp file in same dir, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        # The file fsync above protects bytes; fsync the parent as well so the
        # rename itself survives a sudden notebook/VM loss on POSIX filesystems.
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


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


class JsonlShardWriter:
    """Append-only JSONL shard with periodic flush + atomic replace."""

    def __init__(self, path: str | Path, shard_size: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.records: list[str] = []
        self._existing = self.path.exists()
        if self._existing:
            # resumable: load existing lines into memory ledger
            self.records = [
                ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]

    def append(self, payload: dict[str, Any]) -> int:
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        self.records.append(line)
        return len(self.records)

    def flush(self) -> None:
        atomic_write_text(self.path, "\n".join(self.records) + ("\n" if self.records else ""))

    def __len__(self) -> int:
        return len(self.records)


class RunStore:
    """Manages one run directory with resumable, hashed, append-only output."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.episode_dir = self.run_dir / "episodes"
        self.privileged_dir = self.run_dir / "privileged"
        self.scores_dir = self.run_dir / "monitor_scores"
        self.log_dir = self.run_dir / "logs"
        self.artifact_dir = self.run_dir / "artifacts"
        self.transaction_dir = self.run_dir / "transactions"
        self.commit_dir = self.run_dir / "commits"
        for d in (self.episode_dir, self.privileged_dir, self.scores_dir,
                  self.log_dir, self.artifact_dir, self.transaction_dir, self.commit_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- manifests ----------------------------------------------------------
    def write_run_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.run_dir / "run_manifest.json", manifest)

    def read_run_manifest(self) -> dict[str, Any]:
        path = self.run_dir / "run_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"run manifest missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_plan_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.run_dir / "plan_manifest.json", manifest)

    def read_plan_manifest(self) -> dict[str, Any]:
        path = self.run_dir / "plan_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"plan manifest missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # -- episodes ------------------------------------------------------------
    def completed_episode_ids(self) -> set[str]:
        ledger = self.run_dir / "completed_episode_ids.txt"
        if not ledger.exists():
            return set()
        return {
            ln.strip() for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()
        }

    def record_episode(self, envelope: dict[str, Any], privileged: dict[str, Any],
                       episode_shard: str = "shard-00001.jsonl",
                       privileged_shard: str = "shard-00001.jsonl") -> None:
        """Durably commit the public/privileged pair as one recoverable unit.

        JSONL files cannot be atomically updated together. The write-ahead
        transaction is therefore canonical until a final commit marker exists;
        recovery can safely finish any projection interrupted by a Kaggle VM
        termination without regenerating model output.
        """
        ep_id = envelope["episode_id"]
        if ep_id in self.completed_episode_ids():
            raise ValueError(f"episode {ep_id} already recorded (would duplicate)")
        if privileged.get("episode_id") != ep_id:
            raise ValueError("envelope/privileged episode_id mismatch")
        transaction = {
            "schema_version": "1.0.0",
            "episode_id": ep_id,
            "episode_shard": episode_shard,
            "privileged_shard": privileged_shard,
            "envelope_sha256": sha256_json(envelope),
            "privileged_sha256": sha256_json(privileged),
            "envelope": envelope,
            "privileged": privileged,
        }
        tx_path = self.transaction_dir / f"{ep_id}.json"
        if tx_path.exists():
            existing = json.loads(tx_path.read_text(encoding="utf-8"))
            if sha256_json(existing) != sha256_json(transaction):
                raise ValueError(f"transaction collision for {ep_id}")
        else:
            atomic_write_json(tx_path, transaction)
        self._project_transaction(transaction)

    def _project_transaction(self, transaction: dict[str, Any]) -> None:
        ep_id = transaction["episode_id"]
        self._ensure_jsonl_record(
            self.episode_dir / transaction["episode_shard"],
            transaction["envelope"],
            transaction["envelope_sha256"],
        )
        self._ensure_jsonl_record(
            self.privileged_dir / transaction["privileged_shard"],
            transaction["privileged"],
            transaction["privileged_sha256"],
        )
        ledger_ids = self.completed_episode_ids()
        ledger_ids.add(ep_id)
        atomic_write_text(
            self.run_dir / "completed_episode_ids.txt",
            "".join(f"{item}\n" for item in sorted(ledger_ids)),
        )
        atomic_write_json(
            self.commit_dir / f"{ep_id}.json",
            {
                "schema_version": "1.0.0",
                "episode_id": ep_id,
                "transaction_sha256": sha256_json(transaction),
                "envelope_sha256": transaction["envelope_sha256"],
                "privileged_sha256": transaction["privileged_sha256"],
            },
        )

    @staticmethod
    def _ensure_jsonl_record(path: Path, payload: dict[str, Any], digest: str) -> None:
        writer = JsonlShardWriter(path)
        matches = [
            json.loads(line) for line in writer.records
            if json.loads(line).get("episode_id") == payload["episode_id"]
        ]
        if len(matches) > 1:
            raise ValueError(f"duplicate episode {payload['episode_id']} in {path}")
        if matches:
            if sha256_json(matches[0]) != digest:
                raise ValueError(f"conflicting episode {payload['episode_id']} in {path}")
            return
        writer.append(payload)
        writer.flush()

    def recover_transactions(self) -> dict[str, int]:
        """Finish prepared episode commits without rerunning inference."""
        recovered = 0
        verified = 0
        for tx_path in sorted(self.transaction_dir.glob("*.json")):
            transaction = json.loads(tx_path.read_text(encoding="utf-8"))
            commit_path = self.commit_dir / tx_path.name
            if commit_path.exists():
                commit = json.loads(commit_path.read_text(encoding="utf-8"))
                if commit.get("transaction_sha256") != sha256_json(transaction):
                    raise ValueError(f"commit hash mismatch for {transaction.get('episode_id')}")
                verified += 1
                continue
            self._project_transaction(transaction)
            recovered += 1
        return {"recovered": recovered, "verified_commits": verified}

    # -- monitor scores ------------------------------------------------------
    def score_shard_path(self, monitor_name: str, condition: str, shard: int) -> Path:
        safe = condition.replace(" ", "_")
        return self.scores_dir / f"{monitor_name}-{safe}-{shard:05d}.jsonl"

    def record_scores(self, monitor_name: str, condition: str,
                      score_records: Iterable[dict[str, Any]], shard: int = 1) -> None:
        writer = JsonlShardWriter(self.score_shard_path(monitor_name, condition, shard))
        for rec in score_records:
            writer.append(rec)
        writer.flush()

    # -- integrity ------------------------------------------------------------
    def write_checksums(self) -> Path:
        lines: list[str] = []
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(self.run_dir)}")
        out = self.artifact_dir / "checksums.sha256"
        atomic_write_text(out, "\n".join(lines) + ("\n" if lines else ""))
        return out

    def log(self, name: str, text: str) -> None:
        with open(self.log_dir / name, "a", encoding="utf-8") as handle:
            handle.write(text)

    # -- loading ---------------------------------------------------------------
    def load_all_episodes(self, validate: bool = True) -> list[dict[str, Any]]:
        from .schema import load_episodes

        records: list[dict[str, Any]] = []
        for path in sorted(self.episode_dir.glob("*.jsonl")):
            records.extend(load_episodes(path, validate=validate))
        return records

    def load_all_privileged(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for path in sorted(self.privileged_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    records[rec["episode_id"]] = rec
        return records

    def load_scores(self, monitor_name: str, condition: str) -> list[dict[str, Any]]:
        safe = condition.replace(" ", "_")
        records: list[dict[str, Any]] = []
        for path in sorted(self.scores_dir.glob(f"{monitor_name}-{safe}-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        return records


def envelope_fingerprint(envelope: dict[str, Any]) -> str:
    """Stable hash of the observable content of an episode (for view auditing)."""
    return sha256_json({
        "episode_id": envelope["episode_id"],
        "events": [
            {"type": e["type"], "public": e["public_content"],
             "reasoning": e.get("reasoning_content")}
            for e in envelope["events"]
        ],
    })


__all__ = [
    "RunStore", "JsonlShardWriter", "atomic_write_text", "atomic_write_json",
    "envelope_fingerprint",
]
