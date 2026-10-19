#!/usr/bin/env python3
"""Merge plan-shard run directories into one canonical run directory.

Each generation kernel (shard) writes its own copy of a run directory with
`episodes/shard-00001.jsonl`. This script merges N shard dirs into a single
canonical run dir:

  * episodes/shard-00001.jsonl ... shard-0000N.jsonl  (renumbered per shard)
  * privileged/shard-00001.jsonl ... shard-0000N.jsonl
  * completed_episode_ids.txt                        (disjoint union; duplicates fatal)
  * run_manifest.json / plan_manifest.json           (taken from shard 0; the
    run_id and config hash must be IDENTICAL across shards — a mismatch aborts)
  * checksums.sha256                                  (recomputed by RunStore)

Then runs the full verifier (schema + ledger + outcome re-grade) on the
merged directory. Exits non-zero on any inconsistency. No fallback.

Usage:
    python scripts/merge_shards.py --shard-dir A --shard-dir B --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", action="append", required=True,
                    help="a shard run directory (repeatable, in shard order)")
    ap.add_argument("--out", required=True, help="merged run directory to create")
    args = ap.parse_args()

    shard_dirs = [Path(d).resolve() for d in args.shard_dir]
    out = Path(args.out).resolve()
    for d in shard_dirs:
        if not (d / "run_manifest.json").exists():
            print(f"not a shard run dir (no run_manifest.json): {d}", file=sys.stderr)
            return 2
    if out.exists():
        print(f"refusing to overwrite existing out dir: {out}", file=sys.stderr)
        return 2

    manifests = [json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
                 for d in shard_dirs]
    run_ids = {m.get("run_id") for m in manifests}
    config_hashes = {m.get("config_sha256") for m in manifests}
    if len(run_ids) != 1 or None in run_ids:
        print(f"FATAL: shard run_ids differ: {run_ids}", file=sys.stderr)
        return 1
    if len(config_hashes) != 1:
        print(f"FATAL: shard config hashes differ: {config_hashes}", file=sys.stderr)
        return 1

    plan_manifests = [json.loads((d / "plan_manifest.json").read_text(encoding="utf-8"))
                      for d in shard_dirs]
    if len({json.dumps(pm, sort_keys=True) for pm in plan_manifests}) != 1:
        print("FATAL: plan manifests differ across shards", file=sys.stderr)
        return 1

    out.mkdir(parents=True)
    (out / "episodes").mkdir()
    (out / "privileged").mkdir()
    (out / "transactions").mkdir()
    (out / "commits").mkdir()

    all_episode_ids: list[str] = []
    for i, d in enumerate(shard_dirs, start=1):
        eps = load_jsonl(d / "episodes" / "shard-00001.jsonl")
        priv = load_jsonl(d / "privileged" / "shard-00001.jsonl")
        ep_ids = [e["episode_id"] for e in eps]
        priv_ids = [p["episode_id"] for p in priv]
        if ep_ids != priv_ids:
            print(f"FATAL: shard {i}: episode/privileged id order mismatch", file=sys.stderr)
            return 1
        all_episode_ids.extend(ep_ids)
        with open(out / "episodes" / f"shard-{i:05d}.jsonl", "w", encoding="utf-8") as fh:
            for e in eps:
                fh.write(json.dumps(e, sort_keys=True) + "\n")
        with open(out / "privileged" / f"shard-{i:05d}.jsonl", "w", encoding="utf-8") as fh:
            for p in priv:
                fh.write(json.dumps(p, sort_keys=True) + "\n")
        for episode_id in ep_ids:
            tx_source = d / "transactions" / f"{episode_id}.json"
            commit_source = d / "commits" / f"{episode_id}.json"
            if not tx_source.exists() and not commit_source.exists():
                continue  # legacy pre-v2 shard
            if not tx_source.exists() or not commit_source.exists():
                print(f"FATAL: incomplete transaction pair for {episode_id}", file=sys.stderr)
                return 1
            tx_destination = out / "transactions" / tx_source.name
            commit_destination = out / "commits" / commit_source.name
            if tx_destination.exists() or commit_destination.exists():
                print(f"FATAL: duplicate transaction {episode_id}", file=sys.stderr)
                return 1
            transaction = json.loads(tx_source.read_text(encoding="utf-8"))
            transaction["episode_shard"] = f"shard-{i:05d}.jsonl"
            transaction["privileged_shard"] = f"shard-{i:05d}.jsonl"
            commit = json.loads(commit_source.read_text(encoding="utf-8"))
            commit["transaction_sha256"] = canonical_hash(transaction)
            tx_destination.write_text(
                json.dumps(transaction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            commit_destination.write_text(
                json.dumps(commit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    dupes = {ep for ep in all_episode_ids if all_episode_ids.count(ep) > 1}
    if dupes:
        print(f"FATAL: duplicate episode ids across shards: {sorted(dupes)}", file=sys.stderr)
        return 1
    with open(out / "completed_episode_ids.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(all_episode_ids) + "\n")

    shutil.copy(shard_dirs[0] / "run_manifest.json", out / "run_manifest.json")
    shutil.copy(shard_dirs[0] / "plan_manifest.json", out / "plan_manifest.json")
    merged_manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    merged_manifest["merged_shards"] = [
        {
            "merge_index": index,
            "episode_count": len(load_jsonl(directory / "episodes" / "shard-00001.jsonl")),
            "run_manifest_sha256": hashlib.sha256(
                (directory / "run_manifest.json").read_bytes()
            ).hexdigest(),
        }
        for index, directory in enumerate(shard_dirs, start=1)
    ]
    merged_manifest["merged_shard_count"] = len(shard_dirs)
    (out / "run_manifest.json").write_text(
        json.dumps(merged_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from traceguard.traces.storage import RunStore
    store = RunStore(out)
    store.write_checksums()

    from traceguard.verification import verify_run
    report = verify_run(out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        print("MERGED RUN FAILED VERIFICATION", file=sys.stderr)
        return 1
    print(f"MERGED {len(shard_dirs)} shards, {len(all_episode_ids)} episodes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
