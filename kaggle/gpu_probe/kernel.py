# TRACEGUARD minimal GPU probe - decisive test of current GPU session availability.
# Runs in <60s: cuda check, fp16 matmul, writes /kaggle/working/gpu_probe.json.
# NOT part of the research pipeline - pure infrastructure diagnostics.

import json
import platform
import time

import torch


def main() -> int:
    out = {
        "probe": "traceguard-gpu-probe",
        "ts_utc": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = {
            "name": p.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "vram_gib": round(p.total_memory / 2**30, 2),
        }
        x = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        y = (x @ x).float().sum().item()
        out["fp16_matmul_finite"] = bool(y == y and abs(y) != float("inf"))
        out["matmul_sample"] = y
    with open("/kaggle/working/gpu_probe.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("GPU PROBE COMPLETE", json.dumps(out, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
