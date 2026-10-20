# TRACEGUARD monitoring kernel.
# Scores stored episodes under all configured conditions with the frozen monitor
# prompt + deterministic baselines. Uses the generation kernel's output as its
# input (kernel data source), so monitors see the SAME stored episodes.

import glob
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

TRACEGUARD_COMMIT = "f36d0338c6a6d4f5c6f8e1561723314b7ecbea9e"  # must equal the generation kernel's pinned commit
REPO_URL = "https://github.com/heicfast/traceguard.git"
TORCH_DTYPE = "float16"
MONITORS = ["rules_v1", "keywords_v1", "llm_v1"]

OUT = Path("/kaggle/working")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    started = time.time()
    manifest = {
        "kernel": "traceguard-mon",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pinned_commit": TRACEGUARD_COMMIT,
        "monitors": MONITORS,
    }

    # ---- locate the generation run directory (kernel data source mount) -------
    # Kaggle explodes uploaded dataset zips at the dataset root, so the run dir
    # can appear at any depth under /kaggle/input/<dataset-slug>/... ; match
    # ANY path ending in run/run_manifest.json (recursive).
    candidates = sorted(glob.glob("/kaggle/input/**/run/run_manifest.json", recursive=True))
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
        # OOM, see docs/deviations.md): inference-layer only; registry, views,
        # monitors, grader, schemas, configs untouched.
        "src/traceguard/models/qwen.py",
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
    # Split the monitor model across BOTH T4s (weights ~4+4 GiB): condition C/E
    # prompts carry full agent CoT (pilot traces up to ~61K chars ~= 18K tokens;
    # KV ~= 3.4 GiB at fp16) which OOMs a single 14.56 GiB T4 with the whole
    # 8 GiB model resident (observed 2026-10-20: 256 calls, 20+ OOM failures,
    # all in CoT-bearing conditions). device_map="auto" is an infrastructure
    # change only - the monitor model, prompt, and evidence protocol are
    # unchanged; placement is recorded in the per-score inference audit.
    os.environ["TRACEGUARD_DEVICE_MAP"] = "auto"
    sys.path.insert(0, "/tmp/traceguard/src")

    # ---- run the monitoring CLI over the copied run ------------------------------
    mon_cmd = [sys.executable, "-m", "traceguard.cli", "monitor",
               "--config", config_yaml,
               "--run-dir", str(work_run),
               "--monitors"] + MONITORS
    proc = subprocess.run(mon_cmd, cwd="/tmp/traceguard")
    manifest["monitor_rc"] = proc.returncode

    # ---- copy score artifacts to output ----------------------------------------
    dest = OUT / "traceguard-output" / stage_dir / "run"
    dest.mkdir(parents=True, exist_ok=True)
    for item in ("monitor_scores", "episodes", "privileged", "run_manifest.json",
                 "plan_manifest.json", "logs"):
        src = work_run / item
        if src.exists():
            r = run(["cp", "-r", str(src), str(dest)])
    (dest / "environment_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["kernel_seconds"] = round(time.time() - started, 1)
    (dest / "environment_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MON KERNEL COMPLETE", json.dumps(manifest, indent=2), flush=True)
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
