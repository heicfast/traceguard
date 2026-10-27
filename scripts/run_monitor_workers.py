#!/usr/bin/env python3
"""Score disjoint episode shards on independent model copies.

Two layouts (2026-10-27; docs/deviations.md):

- ``per_gpu`` (default, unchanged behavior): one worker per visible GPU, each
  holding ONE full model copy (``TRACEGUARD_DEVICE_MAP='{"": 0}'`` inside a
  single visible device) and scoring a disjoint episode shard. Max parallel
  throughput for models that fit a single T4 (e.g. the 4B anchor monitor).

- ``single_split``: ONE worker process hosting the model SPLIT ACROSS ALL
  visible GPUs (``TRACEGUARD_DEVICE_MAP=auto``; accelerate balances NF4
  weights ~evenly, e.g. ~4.75 GiB per T4 for a 14B-NF4) and scoring every
  episode sequentially in a single shard. This trades ~2x wall time for the
  >40% per-device memory headroom the 14B monitor cells need: the attempt-2
  per-GPU layout sat at a measured ~15.6K-prompt-char capacity knife edge
  (largest pass 15,659 chars vs smallest fail 15,615 chars) and lost 23/128
  condition-C + 17/128 condition-E calls to genuine capacity exhaustion.

The topology decision lives in :func:`plan_worker_topology` — a pure
function, unit-tested without GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from traceguard.ids import sha256_text
from traceguard.traces.schema import validate_monitor_score
from traceguard.traces.storage import RunStore, atomic_write_text

LAYOUTS = ("per_gpu", "single_split")


def plan_worker_topology(gpu_count: int, layout: str) -> list[dict]:
    """Pure worker-topology planner (no GPU or model access; unit-testable).

    Returns one spec per worker process. Each spec carries the exact
    environment the worker subprocess must run under — device visibility,
    the provider device map, and the episode shard slice — so the launch
    path in :func:`main` is a mechanical translation of these specs.
    """
    if gpu_count <= 0:
        raise ValueError(f"gpu_count must be positive, got {gpu_count}")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown worker layout {layout!r} (expected one of {LAYOUTS})")
    if layout == "per_gpu":
        return [
            {
                "rank": rank,
                "visible_devices": str(rank),
                "device_map": '{"": 0}',
                "shard": rank,
                "shard_count": gpu_count,
            }
            for rank in range(gpu_count)
        ]
    # single_split: exactly one worker; the model spans every visible GPU.
    return [
        {
            "rank": 0,
            "visible_devices": ",".join(str(rank) for rank in range(gpu_count)),
            "device_map": "auto",
            "shard": 0,
            "shard_count": 1,
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--monitors", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--visible-gpus", type=int, default=None)
    parser.add_argument("--layout", choices=LAYOUTS, default="per_gpu")
    args = parser.parse_args()

    if args.visible_gpus is None:
        import torch

        gpu_count = torch.cuda.device_count()
    else:
        gpu_count = args.visible_gpus
    if gpu_count <= 0:
        raise SystemExit("no visible GPUs")
    if args.layout == "single_split" and gpu_count < 2:
        print(json.dumps({
            "ok": False,
            "warning": "single_split requested with fewer than 2 visible GPUs; "
                       "falling back is FORBIDDEN (no-fallback policy) — aborting",
        }, indent=2), file=sys.stderr)
        return 1

    topology = plan_worker_topology(gpu_count, args.layout)

    run_dir = Path(args.run_dir).resolve()
    workers_root = run_dir.parent / f"{run_dir.name}-monitor-workers"
    workers_root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[int, Path, subprocess.Popen[str], TextIO]] = []
    for spec in topology:
        rank = spec["rank"]
        worker_run = workers_root / f"gpu-{rank:02d}" / "run"
        shutil.copytree(run_dir, worker_run)
        log_handle = (worker_run.parent / "monitor.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = spec["visible_devices"]
        env["TRACEGUARD_DEVICE_MAP"] = spec["device_map"]
        command = [
            sys.executable, "-m", "traceguard.cli", "monitor",
            "--config", args.config,
            "--run-dir", str(worker_run),
            "--monitors", *args.monitors,
            "--shard", str(spec["shard"]),
            "--shard-count", str(spec["shard_count"]),
        ]
        if args.conditions:
            command.extend(["--conditions", *args.conditions])
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((rank, worker_run, process, log_handle))

    failures = []
    for rank, _worker_run, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append({"gpu": rank, "return_code": return_code})
    if failures:
        print(json.dumps({"ok": False, "failures": failures}, indent=2), file=sys.stderr)
        return 1

    merged: dict[str, dict[str, dict]] = {}
    for _rank, worker_run, _process, _log in processes:
        for path in sorted((worker_run / "monitor_scores").glob("*.jsonl")):
            target = merged.setdefault(path.name, {})
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_monitor_score(record)
                score_id = record["score_id"]
                previous = target.get(score_id)
                if previous is not None and previous != record:
                    raise RuntimeError(f"conflicting duplicate score_id {score_id}")
                target[score_id] = record
    score_dir = run_dir / "monitor_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in merged.items():
        lines = [json.dumps(records[key], sort_keys=True) for key in sorted(records)]
        atomic_write_text(score_dir / filename, "\n".join(lines) + ("\n" if lines else ""))
    session_dir = run_dir / "artifacts" / "monitor_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    sessions = 0
    for _rank, worker_run, _process, _log in processes:
        for path in sorted((worker_run / "artifacts" / "monitor_sessions").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            claimed = payload.pop("session_sha256")
            computed = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            if claimed != computed:
                raise RuntimeError(f"monitor session hash mismatch: {path}")
            payload["session_sha256"] = claimed
            destination = session_dir / path.name
            if destination.exists() and destination.read_bytes() != path.read_bytes():
                raise RuntimeError(f"conflicting monitor session: {path.name}")
            shutil.copy2(path, destination)
            sessions += 1
    RunStore(run_dir).write_checksums()
    print(json.dumps({
        "ok": True,
        "gpu_workers": len(topology),
        "layout": args.layout,
        "score_files": len(merged),
        "scores": sum(len(records) for records in merged.values()),
        "monitor_sessions": sessions,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
