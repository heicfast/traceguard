"""Fail-closed construction of audited inference providers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from ..ids import sha256_json
from .catalog import validate_model_assignment


def _verify_probe_artifact(spec: Any, role: str, model_id: str) -> None:
    expected = spec.probe_artifact_sha256
    env_key = f"TRACEGUARD_{role.upper()}_PROBE_ARTIFACT"
    path_value = os.environ.get(env_key)
    if not path_value:
        raise RuntimeError(
            f"{model_id} requires its admission artifact at environment variable {env_key}"
        )
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256", None)
    computed = sha256_json(payload)
    if claimed != computed or computed != expected:
        raise RuntimeError(
            f"admission artifact hash mismatch for {model_id}: expected {expected}, "
            f"computed {computed}, claimed {claimed}"
        )
    if payload.get("model_id") != model_id or payload.get("role") != role:
        raise RuntimeError("admission artifact model/role identity mismatch")
    if not payload.get("all_gates_passed"):
        raise RuntimeError("admission artifact contains a failed gate")


def build_hf_provider(
    spec: Any,
    *,
    role: Literal["agent", "monitor", "fact_checker"],
    device_map: str | dict | None = None,
    seed: int = 0,
    allow_unproven: bool = False,
):
    provider_name = spec.provider
    profile = validate_model_assignment(
        spec.model_id,
        role=role,
        quantization=spec.quantization,
        allow_probe_required=bool(spec.probe_artifact_sha256) or allow_unproven,
    )
    if profile.t4_status == "probe_required" and not allow_unproven:
        _verify_probe_artifact(spec, role, spec.model_id)
    if provider_name == "auto":
        provider_name = profile.provider
    if provider_name != profile.provider:
        raise ValueError(
            f"provider {provider_name!r} disagrees with registry value {profile.provider!r}"
        )
    protocol = spec.reasoning_protocol
    if protocol == "auto":
        protocol = profile.reasoning_protocol
    if protocol != profile.reasoning_protocol:
        raise ValueError(
            f"reasoning protocol {protocol!r} disagrees with registry value "
            f"{profile.reasoning_protocol!r}"
        )
    common = {
        "model_id": spec.model_id,
        "revision": None if spec.model_revision.startswith("PIN") else spec.model_revision,
        "tokenizer_revision": (
            None if getattr(spec, "tokenizer_revision", spec.model_revision).startswith("PIN")
            else getattr(spec, "tokenizer_revision", spec.model_revision)
        ),
        "device_map": device_map,
        "torch_dtype": spec.compute_dtype,
        "seed": seed,
        "quantization": spec.quantization,
        "attention_backend": spec.attention_backend,
        "max_context_tokens": spec.max_context_tokens,
        "max_memory_per_gpu_gib": spec.max_memory_per_gpu_gib,
        "adapter_id": spec.adapter_id,
        "adapter_revision": spec.adapter_revision,
    }
    if provider_name == "qwen_hf":
        from .qwen import QwenHFProvider

        provider = QwenHFProvider(**common)
    else:
        if protocol not in {"think_tags", "none"}:
            raise ValueError(f"generic_hf does not support reasoning protocol {protocol!r}")
        from .generic import GenericHFProvider

        provider = GenericHFProvider(**common, reasoning_protocol=protocol)
    provider.model_profile = profile.to_payload()
    provider.probe_artifact_sha256 = spec.probe_artifact_sha256
    return provider


__all__ = ["build_hf_provider"]
