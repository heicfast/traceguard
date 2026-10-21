# TRACEGUARD monitor-cell kernel: cross-monitor-qwen4-deepseek14.
# Re-scores the FROZEN phase-3 anchor cohort (128 stored episodes, agent
# Qwen3-4B) with the DIFFERENT-FAMILY monitor DeepSeek-R1-Distill-Qwen-14B (NF4, think_tags protocol) under all
# five conditions, using the cell config derived verbatim from the generation
# manifest (scripts/derive_monitor_cell_configs.py). Same episodes, same view
# compiler, same frozen prompt: any detection difference is attributable to
# the monitor model, not the evidence.

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
CELL_CONFIG = "configs/phase3_monitor_cells/cross_monitor_deepseek14.yaml"
CELL_ID = "cross-monitor-qwen4-deepseek14"
ANALYZE_MONITOR = "llm_generic_hf_eb379ca5e7"
RELEASE_NAME = "traceguard-phase3-cross-monitor-release.tar.gz"

OUT = Path("/kaggle/working")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    started = time.time()
    manifest = {
        "kernel": "traceguard-mon-cell",
        "cell_id": CELL_ID,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pinned_commit": TRACEGUARD_COMMIT,
        "monitors": MONITORS,
        "cell_config": CELL_CONFIG,
    }

    # ---- locate the generation run directory (kernel data source mount) -------
    # Dataset layouts vary with Kaggle's zip handling: the run manifest may
    # appear either as .../run/run_manifest.json (nested layout) or directly
    # at the dataset root (flat layout). Match any depth, then keep only
    # candidates whose parent directory actually looks like a run (has an
    # episodes/ sibling) so a stray manifest elsewhere can not be picked up.
    candidates = sorted(
        path for path in glob.glob("/kaggle/input/**/run_manifest.json", recursive=True)
        if (Path(path).parent / "episodes").is_dir()
    )
    if not candidates:
        print("NO RUN MANIFEST FOUND. /kaggle/input contents:", flush=True)
        os.system("find /kaggle/input -maxdepth 5 -type d | head -50")
        raise SystemExit(1)
    run_dir = Path(candidates[0]).parent
    stage_dir = run_dir.parent.name
    print("Using run dir:", run_dir, flush=True)
    manifest["run_dir"] = str(run_dir)

    # ---- clone + install pinned code -------------------------------------------
    r = run(["git", "clone", "-q", REPO_URL, "/tmp/traceguard"])
    if r.returncode != 0:
        raise SystemExit(1)
    r = run(["git", "-C", "/tmp/traceguard", "checkout", TRACEGUARD_COMMIT])
    if r.returncode != 0:
        raise SystemExit(1)

    manifest_data = json.loads(Path(candidates[0]).read_text(encoding="utf-8"))
    gen_config = manifest_data.get("config")
    if not gen_config:
        raise SystemExit("run manifest carries no config; cannot verify frozen settings")

    # ---- REVISION CONSISTENCY GATE (ancestor + explicit diff allowlist) -------
    gen_code_revision = gen_config.get("provenance", {}).get("code_revision")
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
        # Inference-layer hotfixes inherited from the anchor scoring path
        # (docs/deviations.md): none of these alter view compilation or the
        # frozen prompt; monitor_gate M1 re-verifies every recorded score hash
        # against exactly this code path.
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
        "src/traceguard/smoke.py",
        ".gitignore",
        "src/traceguard/traces/views.py",
        "src/traceguard/monitors/conditions.py",
        "src/traceguard/monitor_runtime.py",
        # 2026-10-20 matrix freeze (deviation recorded): the frozen study
        # matrix and the monitor-cell configs are DATA — they select which
        # ADMITTED monitor model scores the stored episodes; they cannot alter
        # view compilation, scoring semantics, or the episodes themselves. The
        # runtime additionally enforces that every generation-defining section
        # of the cell config equals the generation manifest byte-for-byte
        # (monitor_runtime consistency gate), so any drift fails loudly.
        "configs/model_matrix_phase3.yaml",
        "configs/phase3_monitor_cells/scale_monitor_qwen14.yaml",
        "configs/phase3_monitor_cells/cross_monitor_deepseek14.yaml",
        # 2026-10-20 owner commit 20a2d6b + stage update: documentation only;
        # no code path imports or executes README content.
        "README.md",
        # 2026-10-20 agent-role probe configs (added after the generation
        # commit, commit 25f5883): consumed ONLY by `traceguard probe-model`
        # (cli.py run_model_probe); verified absent from the monitor/analyze/
        # verify/export import graphs — the scoring path cannot read them.
        "configs/probe/qwen3_14b_agent_probe.yaml",
        "configs/probe/deepseek_r1_14b_agent_probe.yaml",
        # 2026-10-20 fresh-program freeze: the agent-cell configs are data for
        # OTHER cells' generation; model_matrix.py gained freeze-record
        # annotation fields (study-design tooling). Verified not imported by
        # any generation/scoring runtime; never read by this kernel's path.
        "configs/phase3_agent_cells/scale_agent_qwen14.yaml",
        "configs/phase3_agent_cells/cross_agent_deepseek14.yaml",
        "src/traceguard/model_matrix.py",
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

    # ---- CELL CONFIG: frozen selection of the admitted monitor model ----------
    import yaml
    cell_path = Path("/tmp/traceguard") / CELL_CONFIG
    if not cell_path.exists():
        raise SystemExit(f"cell config missing at pinned commit: {CELL_CONFIG}")
    config_payload = yaml.safe_load(cell_path.read_text(encoding="utf-8"))
    # The cell config must reproduce the generation-defining sections and the
    # checked provenance fields EXACTLY (fail early with a clear message; the
    # monitor runtime re-enforces this contract at score time).
    for section in ("seed", "agent", "sandbox", "dataset"):
        if config_payload.get(section) != gen_config.get(section):
            raise SystemExit(
                f"cell config drifts from the frozen generation config in section {section!r}")
    for key in ("code_revision", "experiment_name", "experiment_stage", "replicate_id"):
        if config_payload.get("provenance", {}).get(key) != gen_config.get(
                "provenance", {}).get(key):
            raise SystemExit(f"cell config drifts from frozen provenance field {key!r}")
    if config_payload.get("provenance", {}).get("matrix_cell_id") != CELL_ID:
        raise SystemExit("cell config matrix_cell_id mismatch: wrong cell config for this kernel")
    config_yaml = "/tmp/traceguard-cell-config.yaml"
    Path(config_yaml).write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
    manifest["config_sha256_from_gen_manifest"] = manifest_data.get("config_sha256")
    manifest["cell_config_matrix_cell_id"] = CELL_ID

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

    # NF4 monitor models require bitsandbytes; the anchor 4B (fp16) never
    # needed it, so the anchor kernel pattern lacks this gate (probe kernels
    # have it). Import proves the CUDA kernels are loadable on this image.
    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is None:
        up_bnb = run([sys.executable, "-m", "pip", "install", "-q", "-U",
                      "bitsandbytes>=0.46.1"], check=False)
        if up_bnb.returncode != 0:
            raise SystemExit("bitsandbytes install failed")
    import bitsandbytes  # noqa: F401
    manifest["bitsandbytes_version"] = getattr(bitsandbytes, "__version__", "unknown")

    os.environ["TRACEGUARD_TORCH_DTYPE"] = TORCH_DTYPE
    # Each worker receives one independent model copy and a disjoint episode
    # shard. Worker run directories are private and merged only after schema +
    # score-id validation, avoiding concurrent JSONL writes.
    os.environ["TRACEGUARD_DEVICE_MAP"] = '{"": 0}'
    # Allocator hygiene (2026-10-21): expandable segments eliminate the
    # fragmentation-driven marginal OOMs observed on 14B-NF4-per-T4 sessions.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Admission artifact for the probe-required monitor model: the factory
    # recomputes this file's hash and compares it to the cell config's frozen
    # probe_artifact_sha256 before constructing the provider (fail-closed;
    # no env var -> no model load).
    artifact_rel = "runs/v6/probes/monitor/deepseek_r1_14b_monitor_probe_artifact.json"
    artifact_path = Path("/tmp/traceguard") / artifact_rel
    if not artifact_path.exists():
        raise SystemExit(f"monitor probe artifact missing at pinned commit: {artifact_rel}")
    os.environ["TRACEGUARD_MONITOR_PROBE_ARTIFACT"] = str(artifact_path)
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
        "--monitor", ANALYZE_MONITOR,
    ], cwd="/tmp/traceguard") if proc.returncode == 0 else None
    manifest["analysis_rc"] = analysis_proc.returncode if analysis_proc else None

    # Integrity chain rebuild (2026-10-21): canonicalize AFTER analysis
    # (which writes reports/ + tables/ INTO the run dir), BEFORE verify.
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

    release_archive = OUT / RELEASE_NAME
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
                 "transactions", "commits", "artifacts", "reports"):
        src = work_run / item
        if src.exists():
            r = run(["cp", "-r", str(src), str(dest)])
    (dest / "environment_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["kernel_seconds"] = round(time.time() - started, 1)
    (dest / "environment_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MON CELL KERNEL COMPLETE", json.dumps(manifest, indent=2), flush=True)
    critical = [proc, analysis_proc, canonicalize_proc, verify_proc, export_proc]
    return 0 if all(item is not None and item.returncode == 0 for item in critical) else 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
