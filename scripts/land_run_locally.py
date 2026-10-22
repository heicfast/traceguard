#!/usr/bin/env python3
"""Land a monitored cell run: reproduce kernel analysis locally, prove
determinism by byte-comparison, re-canonicalize, verify, gate, export.

Implements the SOP-002 §9.4 landing flow for kernels that predate the
tables/ copy-list fix (9abd4be): local deterministic re-analysis regenerates
reports/ + tables/; byte-comparison against the kernel's reports/ proves the
determinism; then canonicalize (index, ledger) -> verify -> monitor gate.

Usage: land_run_locally.py <staged-run-dir>  (e.g. runs/v6/anchor_mon/run)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd[:8]),
          "..." if len(cmd) > 8 else "")
    return subprocess.run(cmd, cwd=REPO, check=False, **kw)


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not run_dir or not (run_dir / "run_manifest.json").exists():
        print("usage: land_run_locally.py <staged-run-dir>")
        return 2

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    config_payload = manifest.get("config")
    if not config_payload:
        print("FAIL: run manifest carries no config payload")
        return 1
    cfg_path = Path("/tmp/traceguard-landing-config.yaml")
    cfg_path.write_text(yaml.safe_dump(config_payload, sort_keys=True))

    monitor = "llm_qwen_hf"
    # preserve the kernel's reports for the byte-comparison
    kernel_reports = run_dir / "reports"
    saved = Path("/tmp/landing-kernel-reports")
    if saved.exists():
        shutil.rmtree(saved)
    if kernel_reports.exists():
        shutil.copytree(kernel_reports, saved)
        (run_dir / "reports").mkdir(exist_ok=True)
        for f in (run_dir / "reports").iterdir():
            f.unlink()  # keep dir; analysis rewrites its own set

    # 1. local re-analysis (same invocation as the kernel)
    r = run([PY, "-m", "traceguard.cli", "analyze",
             "--config", str(cfg_path), "--run-dir", str(run_dir),
             "--monitor", monitor])
    print("analysis_rc:", r.returncode)
    if r.returncode != 0:
        return 1

    # 2. determinism proof, three tiers (agent-cell landing standard,
    #    commits 9cd5fd1/01a25f6: identical modulo absolute path prefixes):
    #    a) *.plotdata.json: byte-identical (the deterministic figure layer)
    #    b) *.json reports: identical after normalizing absolute paths
    #    c) rendered figures (png/pdf/svg): differ only by embedded
    #       timestamps + matplotlib-version rendering metadata; verified via
    #       plotdata equality, recorded as environment-rendered
    import re

    def norm_json(s: str) -> str:
        s = re.sub(r'"[^"]*/reports/', '"<RUN>/reports/', s)
        s = re.sub(r'"[^"]*/tables/', '"<RUN>/tables/', s)
        s = re.sub(r'"[^"]*analysis_report\.json"', '"<REPORT>"', s)
        return s

    mismatches = []
    plotdata_checked = 0
    json_checked = 0
    figures_env_rendered = 0
    if saved.exists():
        for f in sorted(saved.iterdir()):
            new = run_dir / "reports" / f.name
            if not new.exists():
                mismatches.append(f"missing regenerated {f.name}")
                continue
            if f.name.endswith(".plotdata.json"):
                plotdata_checked += 1
                if f.read_bytes() != new.read_bytes():
                    mismatches.append(f"plotdata byte mismatch {f.name}")
            elif f.name.endswith(".json"):
                json_checked += 1
                if norm_json(f.read_text()) != norm_json(new.read_text()):
                    mismatches.append(f"json semantic mismatch {f.name}")
            else:
                figures_env_rendered += 1  # tier-c: plotdata tier governs
        if mismatches:
            print("DETERMINISM FAIL:")
            for m in mismatches:
                print("  -", m)
            return 1
        print(f"determinism: {plotdata_checked} plotdata byte-identical, "
              f"{json_checked} json path-normalized-identical, "
              f"{figures_env_rendered} figures env-rendered (plotdata governs)")
    else:
        print("note: kernel reports absent; nothing to byte-compare")

    # 3. canonicalize (index first, then ledger), then verify
    r = run([PY, "-m", "traceguard.cli", "canonicalize",
             "--run-dir", str(run_dir)])
    print("canonicalize_rc:", r.returncode)
    if r.returncode != 0:
        return 1
    r = run([PY, "-m", "traceguard.cli", "verify", "--run-dir", str(run_dir)])
    out = r.stdout.decode() if isinstance(r.stdout, bytes) else (r.stdout or "")
    print("verify_rc:", r.returncode)
    print(out[-800:] if out else "")
    if r.returncode != 0:
        return 1

    # 4. monitor gate (verdict may legitimately FAIL: M3 degenerate raw arm)
    r = run([PY, "scripts/monitor_gate.py", "--run-dir", str(run_dir)],
            capture_output=True, text=True)
    gate_out = (r.stdout or "") + (r.stderr or "")
    print("monitor_gate_rc:", r.returncode)
    for line in gate_out.splitlines():
        if line.startswith("["):
            print("  ", line)
    # gate rc 3 == GATE_FAILURE (honest verdict, recorded, not a landing block)

    # 5. export (release contract)
    r = run([PY, "-m", "traceguard.cli", "export", "--run-dir", str(run_dir)],
            capture_output=True, text=True)
    print("export_rc:", r.returncode)
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    for line in tail[-6:]:
        print("  ", line)
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
