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

TRACEGUARD_COMMIT = "4b9caa0d1f236a0fb2f4a7f6196c62afd6aeec38"  # must equal the generation kernel's pinned commit
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
    candidates = sorted(glob.glob("/kaggle/input/*/traceguard-output/*/run/run_manifest.json"))
    candidates += sorted(glob.glob("/kaggle/input/traceguard-output/*/run/run_manifest.json"))
    if not candidates:
        print("NO RUN MANIFEST FOUND. /kaggle/input contents:", flush=True)
        os.system("find /kaggle/input -maxdepth 4 -type d | head -50")
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
    # time; scoring them with a different code revision silently corrupts
    # every view (e.g. r1 vs r2 goal wording). The monitor code revision MUST
    # match the generation code revision recorded in the run manifest.
    gen_code_revision = config_payload.get("provenance", {}).get("code_revision")
    if not gen_code_revision or str(gen_code_revision).startswith("PIN"):
        raise SystemExit(
            f"generation manifest has no resolved code_revision (got {gen_code_revision!r}); "
            f"refusing to score with unverifiable template registry state")
    if gen_code_revision != TRACEGUARD_COMMIT:
        raise SystemExit(
            f"code revision mismatch: generation ran {gen_code_revision[:10]} but this "
            f"monitor kernel is pinned to {TRACEGUARD_COMMIT[:10]}; repin the monitor "
            f"kernel to the generation commit and re-push")
    manifest["gen_code_revision"] = gen_code_revision

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
    os.environ["TRACEGUARD_DEVICE_MAP"] = '{"": 0}'
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
