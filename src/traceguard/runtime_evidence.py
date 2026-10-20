"""Reproducibility snapshots with secrets excluded by construction."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any

from .ids import sha256_json

SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TRACEGUARD_ATTN_IMPL",
    "TRACEGUARD_DEVICE_MAP",
    "TRACEGUARD_TORCH_DTYPE",
)


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def capture_runtime_environment() -> dict[str, Any]:
    packages = {
        (dist.metadata["Name"] or dist.name): dist.version
        for dist in importlib.metadata.distributions()
    }
    snapshot: dict[str, Any] = {
        "schema_version": "1.0.0",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "safe_environment": {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ},
        "nvidia_smi_query": _command_output([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]),
    }
    try:
        import torch

        snapshot["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        }
    except ImportError:
        snapshot["torch_runtime"] = None
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    return snapshot


__all__ = ["capture_runtime_environment"]
