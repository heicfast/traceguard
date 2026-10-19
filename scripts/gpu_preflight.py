#!/usr/bin/env python3
"""TRACEGUARD hardware/torch preflight. Requires PyTorch; does not download models.

Run on Kaggle before installing quantized inference. Returns nonzero if a GPU is
present but fails actual matmul/reduction kernels. Output is a factual probe,
not an estimate that Qwen3 will fit in VRAM.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch unavailable: this preflight requires torch", file=sys.stderr)
        return 2

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
        "check": "skipped_no_gpu",
    }
    if torch.cuda.is_available():
        try:
            for ix in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(ix)
                device = torch.device(f"cuda:{ix}")
                t = torch.randn((512, 512), device=device, dtype=torch.float16)
                value = ((t @ t).float().sum()).item()
                torch.cuda.synchronize(device)
                manifest["devices"].append({
                    "index": ix,
                    "name": prop.name,
                    "capability": list(torch.cuda.get_device_capability(ix)),
                    "vram_gib": round(prop.total_memory / 2**30, 3),
                    "matmul_reduction_finite": value == value and abs(value) != float("inf"),
                })
            manifest["check"] = (
                "passed" if all(x["matmul_reduction_finite"] for x in manifest["devices"])
                else "failed_nonfinite_output"
            )
        except Exception as exc:  # diagnostics must be retained and exit nonzero
            manifest["check"] = "failed_runtime"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10, check=False)
        if output.returncode == 0:
            manifest["nvidia_smi"] = output.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    outdir = Path(os.environ.get("TRACEGUARD_PREFLIGHT_DIR", "/kaggle/working/traceguard-preflight"))
    if not outdir.parent.exists():
        outdir = Path.cwd() / "traceguard-preflight"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "environment.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Saved:", path)
    return 1 if manifest["check"].startswith("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
