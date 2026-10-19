# TRACEGUARD GPU preflight kernel.
# Runs BEFORE any model work: real tensor kernels, environment capture, disk/CPU probe.
# No model downloads. Exits nonzero on failure (Kaggle marks the run as ERROR).

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

TRACEGUARD_COMMIT = "aac912e6b14cbcc3a1d9385f573cc15faad58c8d"
REPO_URL = "https://github.com/heicfast/traceguard.git"
OUT = Path("/kaggle/working/traceguard-preflight")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kernel": "traceguard-preflight",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pinned_commit": TRACEGUARD_COMMIT,
    }

    # ---- disk ---------------------------------------------------------------
    total, used, free = shutil.disk_usage("/kaggle/working")
    manifest["disk_working_gib"] = {"total": round(total / 2**30, 1),
                                     "used": round(used / 2**30, 1),
                                     "free": round(free / 2**30, 1)}

    # ---- package versions (before any upgrade) --------------------------------
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda_build"] = torch.version.cuda
    except ImportError:
        manifest["torch"] = None
    try:
        import transformers
        manifest["transformers"] = transformers.__version__
    except ImportError:
        manifest["transformers"] = None
    try:
        import accelerate
        manifest["accelerate"] = accelerate.__version__
    except ImportError:
        manifest["accelerate"] = None

    # ---- GPU probe with REAL kernels ------------------------------------------
    check = "skipped_no_torch"
    if manifest.get("torch"):
        import torch
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        manifest["devices"] = []
        if torch.cuda.is_available():
            try:
                for ix in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(ix)
                    device = torch.device(f"cuda:{ix}")
                    x = torch.randn((512, 512), device=device, dtype=torch.float16)
                    y = (x @ x).float().sum()
                    torch.cuda.synchronize(device)
                    manifest["devices"].append({
                        "index": ix,
                        "name": prop.name,
                        "capability": list(torch.cuda.get_device_capability(ix)),
                        "vram_gib": round(prop.total_memory / 2**30, 3),
                        "fp16_matmul_reduction_finite": (y.item() == y.item()
                                                         and abs(y.item()) != float("inf")),
                    })
                    # bf16 support probe (T4 sm_75 does not support bf16 natively)
                    try:
                        b = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
                        (b @ b).float().sum().item()
                        bf16_ok = True
                    except Exception:
                        bf16_ok = False
                    manifest["devices"][-1]["bf16_usable"] = bf16_ok
                check = ("passed" if all(d["fp16_matmul_reduction_finite"]
                                          for d in manifest["devices"])
                         else "failed_nonfinite_output")
            except Exception as exc:
                check = "failed_runtime"
                manifest["error"] = f"{type(exc).__name__}: {exc}"

        # CPU fallback determinism spot-check (engine is CPU-first)
        torch.manual_seed(0)
        a = torch.randn(256, 256)
        torch.manual_seed(0)
        b = torch.randn(256, 256)
        manifest["cpu_rng_deterministic"] = bool(torch.equal(a, b))
    manifest["check"] = check

    # ---- nvidia-smi -------------------------------------------------------------
    smi = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader"])
    if smi.returncode == 0:
        manifest["nvidia_smi"] = smi.stdout.strip()

    # ---- clone pinned repo + doctor (research core must run anywhere) ----
    clone = run(["git", "clone", "-q", REPO_URL, "traceguard"])
    if clone.returncode == 0:
        co = run(["git", "-C", "traceguard", "checkout", TRACEGUARD_COMMIT])
        manifest["repo_checkout"] = "ok" if co.returncode == 0 else f"fail:{co.stderr[:200]}"
        if co.returncode == 0:
            inst = run([sys.executable, "-m", "pip", "install", "-q", "-e",
                        "traceguard[analysis]"], check=False)
            manifest["pip_install_rc"] = inst.returncode
            doc = run([sys.executable, "-m", "traceguard.cli", "doctor"], cwd="traceguard")
            manifest["doctor_ok"] = '"template_validation_ok": true' in doc.stdout
            manifest["doctor_excerpt"] = doc.stdout[:1500]
    else:
        manifest["repo_checkout"] = f"clone_failed:{clone.stderr[:200]}"

    (OUT / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Saved:", OUT / "environment.json")
    return 0 if check == "passed" and manifest.get("doctor_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
