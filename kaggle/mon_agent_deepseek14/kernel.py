# TRACEGUARD agent-cohort monitoring kernel: cross-agent-deepseek14-qwen4.
# Scores the 64-episode Qwen3-14B-agent cohort with the FROZEN 4B anchor
# monitor (the weak-to-strong stress axis). Same anchor-kernel pattern: config
# derived from the cohort's own frozen generation manifest; the cohort config
# carries matrix_cell_id, so monitor identities are suffixed
# (llm_qwen_hf_46be2e470c) and can never collide with anchor-cell scores.

import glob
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

TRACEGUARD_COMMIT = "1394950e509d7699e42c62cc152eb193812f2fbd"  # instrument v2 + chain order + allocator hygiene (attempt-1 fixes)
REPO_URL = "https://github.com/heicfast/traceguard.git"
TORCH_DTYPE = "float16"
MONITORS = ["rules_v1", "keywords_v1", "llm_v1", "late_fusion"]

OUT = Path("/kaggle/working")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    started = time.time()
    manifest = {
        "kernel": "traceguard-mon-agent-cohort",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pinned_commit": TRACEGUARD_COMMIT,
        "monitors": MONITORS,
    }

    # ---- locate the generation run directory (kernel data source mount) -------
    # Kaggle explodes uploaded dataset zips at the dataset root, so the run dir
    # can appear at any depth under /kaggle/input/<dataset-slug>/... , in two
    # layouts: wrapped (.../<any>/run/run_manifest.json) or FLAT (the zip root
    # WAS the run dir: .../<slug>/run_manifest.json with episodes/ etc. as
    # siblings). Match any run_manifest.json recursively, then keep only
    # candidates whose parent directory actually looks like a run dir (has an
    # episodes/ sibling) — robust to both layouts (2026-10-21: the agent
    # cohort datasets mount flat and the wrapped-only glob missed them).
    candidates = sorted(
        p for p in glob.glob("/kaggle/input/**/run_manifest.json", recursive=True)
        if (Path(p).parent / "episodes").is_dir()
    )
    if not candidates:
        print("NO RUN MANIFEST FOUND. /kaggle/input contents:", flush=True)
        os.system("find /kaggle/input -maxdepth 5 -type d | head -50")
        raise SystemExit(1)
    run_dir = Path(candidates[0]).parent
    # Wrapped layout: .../<stage_dir>/run  -> stage_dir = parent dir name.
    # Flat layout: .../<slug>              -> use the manifest's stage field.
    if run_dir.name == "run":
        stage_dir = run_dir.parent.name
    else:
        _rm = json.loads(Path(candidates[0]).read_text(encoding="utf-8"))
        stage_dir = _rm.get("stage") or _rm.get("experiment_stage") or "pilot"
    print("Using run dir:", run_dir, flush=True)
    manifest["run_dir"] = str(run_dir)

    # ---- clone + install pinned code -------------------------------------------
    r = run(["git", "clone", "-q", REPO_URL, "/tmp/traceguard"])
    if r.returncode != 0:
        raise SystemExit(1)
    r = run(["git", "-C", "/tmp/traceguard", "checkout", TRACEGUARD_COMMIT])
    if r.returncode != 0:
        raise SystemExit(1)

    # derive the exact config from the generation run's frozen manifest so the
    # monitor stage uses the same frozen settings as the generation stage
    manifest_data = json.loads(Path(candidates[0]).read_text(encoding="utf-8"))
    config_payload = manifest_data.get("config")
    if not config_payload:
        raise SystemExit("run manifest carries no config; cannot reproduce frozen settings")

    # ---- REVISION CONSISTENCY GATE -------------------------------------------
    # The episodes embed task instances from the code registry at generation
    # time; scoring them with a changed template registry / view compiler /
    # monitor definition would silently corrupt every view. The monitor commit
    # must therefore be a DESCENDANT of the generation commit AND the diff
    # between them may touch ONLY the explicitly allowlisted inference-layer
    # files (recorded in docs/deviations.md per hotfix). Equality still passes.
    gen_code_revision = config_payload.get("provenance", {}).get("code_revision")
    if not gen_code_revision or str(gen_code_revision).startswith("PIN"):
        raise SystemExit(
            f"generation manifest has no resolved code_revision (got {gen_code_revision!r}); "
            f"refusing to score with unverifiable template registry state")
    anc = run(["git", "-C", "/tmp/traceguard", "merge-base", "--is-ancestor",
               gen_code_revision, TRACEGUARD_COMMIT])
    if anc.returncode != 0:
        raise SystemExit(
            f"generation commit {gen_code_revision[:10]} is not an ancestor of the "
            f"monitor pin {TRACEGUARD_COMMIT[:10]}; refusing to score across a fork")
    diff = run(["git", "-C", "/tmp/traceguard", "diff", "--name-only",
                gen_code_revision, TRACEGUARD_COMMIT])
    allowed_changes = {
        # 2026-10-20 prefill-logits guard + device_map hotfix (monitor-kernel
        # OOM, see docs/deviations.md): inference-layer only.
        "src/traceguard/models/qwen.py",
        # 2026-10-21 instrument v2 (docs/deviations.md): the monitor reports
        # rubric components only and the parser derives the score; the
        # fusion adjudicator is treated identically. llm.py/structured.py
        # carry the v2 prompts + derived-score parsers; verification.py
        # carries the frozen-prompt registry with the attempt-1 legacy
        # hashes (so the attempt-1 archive keeps verifying); cli.py adds
        # the canonicalize command (index, then ledger covering the index)
        # which this kernel now runs between analyze and verify.
        "src/traceguard/monitors/llm.py",
        "src/traceguard/monitors/structured.py",
        "src/traceguard/verification.py",
        "src/traceguard/cli.py",
        # 2026-10-20 smoke-stage E-pairing fix: smoke.py is integration
        # validation ONLY (imported function-locally by cmd_smoke; never by
        # monitor_runtime/cli monitor path) - verified not in the scoring
        # import graph, so it cannot affect how stored episodes are scored.
        "src/traceguard/smoke.py",
        # 2026-10-20 evidence-dir whitelist edit: zero runtime semantics.
        ".gitignore",
        # 2026-10-20 PREDECLARED EVIDENCE WINDOW (docs/deviations.md): the
        # reasoning section of every CoT-bearing view is head-capped at
        # 12,288 chars (config max_monitor_tokens 3072 x 4.0 chars/token),
        # symmetrically across C/D/E/F, declared BEFORE any admissible LLM
        # scores existed. The windowed views ARE the monitor's evidence
        # definition for this stage; monitor_gate M1 re-verifies every
        # recorded score hash against exactly this code path.
        "src/traceguard/traces/views.py",
        "src/traceguard/monitors/conditions.py",
        "src/traceguard/monitor_runtime.py",
    }
    non_research_prefixes = ("docs/", "scripts/", "kaggle/", "tests/", "runs/")
    changed = [ln for ln in diff.stdout.splitlines() if ln.strip()]
    disallowed = [ln for ln in changed
                  if ln not in allowed_changes and not ln.startswith(non_research_prefixes)]
    if disallowed:
        raise SystemExit(
            f"research-core files changed between generation {gen_code_revision[:10]} and "
            f"monitor pin {TRACEGUARD_COMMIT[:10]}: {disallowed} — regeneration or a new "
            f"deviation-approved allowlist entry is required")
    manifest["gen_code_revision"] = gen_code_revision
    manifest["revision_gate"] = {
        "mode": "ancestor+diff-allowlist",
        "changed_files": changed,
        "allowed_hotfix_files": sorted(allowed_changes),
    }

    import yaml
    config_yaml = "/tmp/traceguard-config-from-manifest.yaml"
    Path(config_yaml).write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
    manifest["config_sha256_from_gen_manifest"] = manifest_data.get("config_sha256")

    # copy run artifacts to a writable location (input mounts are read-only)
    work_run = Path("/kaggle/working/mon-input") / stage_dir / "run"
    work_run.parent.mkdir(parents=True, exist_ok=True)
    r = run(["cp", "-r", str(run_dir), str(work_run)])
    if r.returncode != 0:
        raise SystemExit(1)

    r = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
             "/tmp/traceguard[analysis]"], check=False)
    manifest["pip_rc"] = r.returncode
    import importlib

    import transformers
    manifest["transformers_before"] = transformers.__version__
    from packaging.version import parse as vparse
    if vparse(transformers.__version__) < vparse("4.51"):
        up = run([sys.executable, "-m", "pip", "install", "-q", "-U",
                  "transformers>=4.51,<5"], check=False)
        if up.returncode != 0:
            raise SystemExit(1)
        importlib.invalidate_caches()
    manifest["transformers_used"] = transformers.__version__

    os.environ["TRACEGUARD_TORCH_DTYPE"] = TORCH_DTYPE
    # Each worker receives one independent model copy and a disjoint episode
    # shard. Worker run directories are private and merged only after schema +
    # score-id validation, avoiding concurrent JSONL writes.
    os.environ["TRACEGUARD_DEVICE_MAP"] = '{"": 0}'
    # Allocator hygiene (2026-10-21) — same fragmentation fix as the fleet.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/tmp/traceguard/src")

    # ---- run the monitoring CLI over the copied run ------------------------------
    mon_cmd = [sys.executable, "/tmp/traceguard/scripts/run_monitor_workers.py",
               "--config", config_yaml,
               "--run-dir", str(work_run),
               "--monitors"] + MONITORS
    proc = subprocess.run(mon_cmd, cwd="/tmp/traceguard")
    manifest["monitor_rc"] = proc.returncode

    # Analysis and integrity release run on the exact scored copy. Pilot design
    # gates may correctly fail without discarding artifacts; critical pipeline
    # commands remain binding.
    analysis_proc = subprocess.run([
        sys.executable, "-m", "traceguard.cli", "analyze",
        "--config", config_yaml, "--run-dir", str(work_run),
        "--monitor", "llm_qwen_hf_46be2e470c",
    ], cwd="/tmp/traceguard") if proc.returncode == 0 else None
    manifest["analysis_rc"] = analysis_proc.returncode if analysis_proc else None

    # Integrity chain rebuild (2026-10-21): canonicalize AFTER analysis,
    # BEFORE verify (index first, then the ledger covering the index).
    canonicalize_proc = subprocess.run([
        sys.executable, "-m", "traceguard.cli", "canonicalize",
        "--run-dir", str(work_run),
    ], cwd="/tmp/traceguard") if analysis_proc and analysis_proc.returncode == 0 else None
    manifest["canonicalize_rc"] = (
        canonicalize_proc.returncode if canonicalize_proc else None)

    gate_proc = subprocess.run([
        sys.executable, "/tmp/traceguard/scripts/monitor_gate.py",
        "--run-dir", str(work_run),
        "--calibration-split", config_payload["monitor"]["calibration_split"],
    ], cwd="/tmp/traceguard") if proc.returncode == 0 else None
    manifest["monitor_gate_rc"] = gate_proc.returncode if gate_proc else None

    verify_proc = subprocess.run([
        sys.executable, "-m", "traceguard.cli", "verify", "--run-dir", str(work_run),
    ], cwd="/tmp/traceguard") if proc.returncode == 0 else None
    manifest["verify_rc"] = verify_proc.returncode if verify_proc else None

    release_archive = OUT / "traceguard-phase3-agent-deepseek14-mon-release.tar.gz"
    export_proc = subprocess.run([
        sys.executable, "-m", "traceguard.cli", "export",
        "--run-dir", str(work_run), "--archive", str(release_archive),
    ], cwd="/tmp/traceguard") if (analysis_proc and analysis_proc.returncode == 0
                                     and canonicalize_proc
                                     and canonicalize_proc.returncode == 0) else None
    manifest["export_rc"] = export_proc.returncode if export_proc else None

    # ---- copy score artifacts to output ----------------------------------------
    dest = OUT / "traceguard-output" / stage_dir / "run"
    dest.mkdir(parents=True, exist_ok=True)
    for item in ("monitor_scores", "episodes", "privileged", "run_manifest.json",
                 "plan_manifest.json", "logs", "completed_episode_ids.txt",
                 "transactions", "commits", "artifacts", "reports", "tables"):
        src = work_run / item
        if src.exists():
            r = run(["cp", "-r", str(src), str(dest)])
    (dest / "environment_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["kernel_seconds"] = round(time.time() - started, 1)
    (dest / "environment_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MON KERNEL COMPLETE", json.dumps(manifest, indent=2), flush=True)
    critical = [proc, analysis_proc, canonicalize_proc, verify_proc, export_proc]
    return 0 if all(item is not None and item.returncode == 0 for item in critical) else 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
