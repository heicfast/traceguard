# TRACEGUARD split-layout admission probe kernel (DeepSeek-R1-14B monitor).
# Validates the CROSS-GPU NF4 load path (device_map=auto across BOTH T4s)
# that the cross-monitor cell will use, INCLUDING the deployment-scale
# stress gate (~17.9K-char evidence-window prompt) that admission previously
# lacked — the gap that let the attempt-2 cells reach a capacity knife edge
# no probe had ever exercised (docs/deviations.md, 2026-10-27). Fail-closed
# on any gate; the artifact hash is what the cell config pins.

import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ---------------- run parameters (per candidate directory) ---------------------
TRACEGUARD_COMMIT = "PIN_EXACT_COMMIT_BEFORE_RUN"
REPO_URL = "https://github.com/heicfast/traceguard.git"
CONFIG = "configs/probe/deepseek_r1_14b_monitor_probe.yaml"
ROLE = "monitor"
CANDIDATE = "deepseek_r1_14b_split"

OUT = Path("/kaggle/working")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gate(condition: bool, message: str, manifest: dict) -> None:
    manifest.setdefault("gates", []).append(
        {"gate": message, "passed": bool(condition),
         "time_utc": datetime.now(UTC).isoformat()})
    if not condition:
        (OUT / "kernel_failure.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("GATE FAILED:", message, flush=True)
        raise SystemExit(1)
    print("GATE PASSED:", message, flush=True)


def main() -> int:
    started = time.time()
    manifest = {
        "kernel": f"traceguard-probe-{CANDIDATE}",
        "candidate": CANDIDATE,
        "config": CONFIG,
        "role": ROLE,
        "layout": "split",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "pinned_commit": TRACEGUARD_COMMIT,
        "gates": [],
    }

    import torch
    gate(torch.cuda.is_available(), "cuda available", manifest)
    # The split layout REQUIRES both T4s: device_map=auto across one visible
    # device is not the deployment path.
    gate(torch.cuda.device_count() >= 2,
         f"at least two gpus for the split layout (saw {torch.cuda.device_count()})",
         manifest)
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        manifest.setdefault("gpus", []).append({
            "index": index, "name": props.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_gib": round(props.total_memory / 2**30, 2)})
    print("GPUs:", manifest["gpus"], flush=True)

    # ---- repo clone at pinned commit -------------------------------------------
    r = run(["git", "clone", "-q", REPO_URL, "/tmp/traceguard"])
    gate(r.returncode == 0, "clone repository", manifest)
    r = run(["git", "-C", "/tmp/traceguard", "checkout", TRACEGUARD_COMMIT])
    gate(r.returncode == 0, f"checkout {TRACEGUARD_COMMIT[:10]}", manifest)

    # ---- install: repo (no deps) + bitsandbytes for NF4 ------------------------
    r = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
             "/tmp/traceguard"], check=False)
    manifest["pip_repo_rc"] = r.returncode
    gate(r.returncode == 0, "install traceguard package", manifest)

    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is None:
        r = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                check=False)
        manifest["pip_bnb_rc"] = r.returncode
        gate(r.returncode == 0, "install bitsandbytes", manifest)
    import bitsandbytes  # noqa: F401  (import proves CUDA kernels loadable below)
    manifest["bitsandbytes_version"] = getattr(bitsandbytes, "__version__", "unknown")
    print("bitsandbytes:", manifest["bitsandbytes_version"], flush=True)

    import transformers
    manifest["transformers_before"] = transformers.__version__
    from packaging.version import parse as vparse
    if vparse(transformers.__version__) < vparse("4.51"):
        r = run([sys.executable, "-m", "pip", "install", "-q", "-U",
                 "transformers>=4.51"], check=False)
        manifest["pip_transformers_rc"] = r.returncode
        gate(r.returncode == 0, "upgrade transformers >=4.51", manifest)
        import importlib
        importlib.invalidate_caches()
        import transformers
    manifest["transformers_used"] = transformers.__version__
    print("transformers:", manifest["transformers_used"], flush=True)

    # ---- run the admission probe on the SPLIT load path ------------------------
    artifact_path = OUT / f"probe_artifact_{CANDIDATE}.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/tmp/traceguard/src"
    # THE LOAD PATH UNDER TEST: accelerate balances the NF4 weights across
    # every visible device. This is exactly what the single_split worker
    # layout sets for the scale cell.
    env["TRACEGUARD_DEVICE_MAP"] = "auto"
    manifest["device_map"] = "auto"
    proc = subprocess.run(
        [sys.executable, "-m", "traceguard.cli", "probe-model",
         "--config", CONFIG, "--role", ROLE, "--out", str(artifact_path)],
        cwd="/tmp/traceguard", env=env)
    manifest["probe_rc"] = proc.returncode
    gate(proc.returncode == 0, f"probe gates passed ({CANDIDATE}, split layout)", manifest)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = artifact.get("artifact_sha256")
    manifest["resolved_model_revision"] = artifact.get("model_revision")
    manifest["resolved_tokenizer_revision"] = artifact.get("tokenizer_revision")
    manifest["probe_gates"] = artifact.get("gates")
    manifest["stress_prompt_metrics"] = artifact.get("stress_prompt_metrics")
    manifest["probe_schema_version"] = artifact.get("schema_version")
    manifest["kernel_seconds"] = round(time.time() - started, 1)

    (OUT / f"probe_environment_{CANDIDATE}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PROBE KERNEL COMPLETE", json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
