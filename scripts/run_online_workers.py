#!/usr/bin/env python3
"""Run one independent TRACEGUARD online-study worker per visible GPU.

Same audited data-parallel topology as run_gpu_workers.py (workers never
share output files; verified run directories are merged only after every
process exits successfully), dispatching the RQ4 ``traceguard online``
driver instead of generation. The online driver loads ONE shared 4B
checkpoint per worker serving both the agent and the monitor arm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpu_workers import WorkerSpec, worker_specs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel-shard", type=int, required=True)
    parser.add_argument("--kernel-shard-count", type=int, required=True)
    parser.add_argument("--visible-gpus", type=int, help="test override")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.visible_gpus is not None:
        gpu_count = args.visible_gpus
    else:
        import torch

        gpu_count = torch.cuda.device_count()
    specs: list[WorkerSpec] = worker_specs(gpu_count, args.kernel_shard, args.kernel_shard_count)
    if args.dry_run:
        print(json.dumps([s.__dict__ for s in specs], indent=2, sort_keys=True))
        return 0

    output = Path(args.output).resolve()
    workers_dir = output / "workers"
    workers_dir.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[WorkerSpec, subprocess.Popen[str], TextIO]] = []
    for spec in specs:
        worker_root = workers_dir / f"gpu-{spec.local_rank:02d}"
        worker_root.mkdir(parents=True)
        log_handle = (worker_root / "online.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec.physical_gpu)
        env["TRACEGUARD_DEVICE_MAP"] = '{"": 0}'
        command = [
            sys.executable,
            "-m",
            "traceguard.cli",
            "online",
            "--config",
            args.config,
            "--run-dir",
            str(worker_root / "run"),
            "--shard",
            str(spec.global_shard),
            "--shard-count",
            str(spec.global_shard_count),
        ]
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((spec, process, log_handle))

    failures: list[dict[str, int]] = []
    for spec, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append({"local_rank": spec.local_rank, "return_code": return_code})
    if failures:
        print(json.dumps({"ok": False, "worker_failures": failures}, indent=2), file=sys.stderr)
        return 1

    merge_command = [sys.executable, str(Path(__file__).with_name("merge_shards.py"))]
    for spec in specs:
        merge_command.extend(
            ["--shard-dir", str(workers_dir / f"gpu-{spec.local_rank:02d}" / "run")]
        )
    merge_command.extend(["--out", str(output / "run")])
    merged = subprocess.run(merge_command, check=False)
    if merged.returncode != 0:
        return merged.returncode
    print(
        json.dumps(
            {"ok": True, "workers": [s.__dict__ for s in specs], "run": str(output / "run")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
