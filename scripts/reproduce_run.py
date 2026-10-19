#!/usr/bin/env python3
"""One-command reproduction of the primary analysis from archived artifacts.

Usage:
    python scripts/reproduce_run.py --run-dir runs/<stage>/run --monitor llm_qwen_hf

This re-validates the run (schema, ledger, outcome re-grade), re-runs the
analysis, and re-emits the primary figures and report from stored evidence
only. It must produce byte-identical report numbers on the same artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--monitor", default="llm_qwen_hf")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "run_manifest.json").exists():
        print(f"no run_manifest.json under {run_dir}", file=sys.stderr)
        return 2

    # 1. verification (schema + ledger + outcome re-grade + checksums)
    verify = subprocess.run(
        [sys.executable, "-m", "traceguard.cli", "verify", "--run-dir", str(run_dir)],
        capture_output=True, text=True, cwd=str(REPO))
    print(verify.stdout)
    if verify.returncode != 0:
        print("VERIFICATION FAILED — refusing to analyze", file=sys.stderr)
        return 1

    # 2. locate the frozen config: prefer the run manifest's embedded config
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config_path = run_dir.parent / "config_as_run.yaml"
    if "config" in manifest:
        import yaml
        config_path.write_text(
            yaml.safe_dump(manifest["config"], sort_keys=True), encoding="utf-8")
    else:
        print("run manifest lacks embedded config; pass --config manually",
              file=sys.stderr)
        return 2

    # 3. analysis
    out = args.out or str(run_dir / "reports")
    analysis = subprocess.run(
        [sys.executable, "-m", "traceguard.cli", "analyze",
         "--config", str(config_path), "--run-dir", str(run_dir),
         "--out", out, "--monitor", args.monitor],
        capture_output=True, text=True, cwd=str(REPO))
    print(analysis.stdout)
    print(analysis.stderr[-3000:] if analysis.stderr else "", file=sys.stderr)
    return analysis.returncode


if __name__ == "__main__":
    raise SystemExit(main())
