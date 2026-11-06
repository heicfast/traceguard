"""Fail-closed construction of audited inference providers.

v0.7.0 adds the serving-plane construction path: configs whose provider is
``vllm_openai`` or ``api_openai`` build an
:class:`traceguard.serving.openai_compat.OpenAICompatProvider` against the
pinned endpoint, after the same audited-registry admission checks (hardware
tier aware).  The transformers path is unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from ..ids import sha256_json
from .catalog import validate_model_assignment

ENDPOINT_PROVIDERS = ("vllm_openai", "api_openai")


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
    tier = getattr(spec, "hardware_tier", "kaggle_t4")
    if payload.get("hardware_tier", "kaggle_t4") != tier:
        raise RuntimeError(
            f"admission artifact was recorded on tier {payload.get('hardware_tier')!r} "
            f"but the config requests {tier!r}; re-probe on the target tier"
        )


def _endpoint_provider_kwargs(spec: Any) -> dict[str, Any]:
    return {
        "api_base_url": getattr(spec, "api_base_url", None),
        "api_model_id": getattr(spec, "api_model_id", None),
        "api_key_env": getattr(spec, "api_key_env", "TRACEGUARD_API_KEY"),
    }


def build_endpoint_provider(
    spec: Any,
    *,
    role: Literal["agent", "monitor", "fact_checker"],
    cost_ledger: Any | None = None,
) -> Any:
    """Build an OpenAI-compatible provider (vLLM server or remote API).

    Admission follows the same audited registry as the transformers path,
    evaluated on the config's hardware tier.  The probe artifact must have
    been recorded on the SAME tier.  Determinism and pricing are recorded on
    every model turn / ledger event.
    """
    from ..serving.cost import PriceTable, price_for
    from ..serving.openai_compat import OpenAICompatProvider, ProviderEndpoint

    provider_name = spec.provider
    profile = validate_model_assignment(
        spec.model_id,
        role=role,
        quantization=spec.quantization,
        allow_probe_required=bool(spec.probe_artifact_sha256),
        hardware_tier=getattr(spec, "hardware_tier", "kaggle_t4"),
    )
    if provider_name == "auto":
        provider_name = (
            "api_openai" if profile.kind == "remote_api" else "vllm_openai"
        )
    # Kind compatibility: vllm_openai serves LOCAL weights through a pinned
    # server; api_openai is exclusively for remote_api registry entries.
    # (The registry's `provider` field describes the TRANSFORMERS adapter and
    # does not constrain which server the checkpoint runs on.)
    if provider_name == "api_openai" and profile.kind != "remote_api":
        raise ValueError(
            f"{spec.model_id} is a local-weights model; remote-API construction "
            "is reserved for registry entries of kind remote_api"
        )
    if provider_name == "vllm_openai" and profile.kind != "local_weights":
        raise ValueError(
            f"{spec.model_id} is a remote API model; serve it with api_openai"
        )
    kwargs = _endpoint_provider_kwargs(spec)
    if not kwargs["api_base_url"] or not kwargs["api_model_id"]:
        raise ValueError(
            "endpoint providers require api_base_url and api_model_id in the config"
        )
    # vLLM servers parse reasoning server-side (reasoning_content); remote
    # APIs vary — the admission probe pins this per endpoint, default on for
    # local servers, on for the DeepSeek-style APIs that document the field.
    supports_reasoning = profile.reasoning_protocol in ("qwen3", "think_tags",
                                                         "reasoning_field",
                                                         "harmony_channels")
    endpoint = ProviderEndpoint(
        base_url=kwargs["api_base_url"],
        model_id=kwargs["api_model_id"],
        api_key_env=kwargs["api_key_env"],
        provider_name=provider_name,
        supports_seed=(provider_name == "vllm_openai"),
        supports_reasoning_field=supports_reasoning,
        determinism_class=(
            "seeded_server_best_effort" if provider_name == "vllm_openai"
            else "unseeded_api"
        ),
    )
    if provider_name == "vllm_openai" and not str(endpoint.base_url).startswith(
        ("http://127.0.0.1", "http://localhost")
    ):
        raise ValueError(
            "vllm_openai endpoints must point at a local server (127.0.0.1/localhost); "
            "remote APIs use the api_openai provider"
        )
    price_table = PriceTable()
    price = price_for(
        price_table,
        "openai" if provider_name == "api_openai" else "vllm",
        kwargs["api_model_id"],
        (None, None),
    )
    provider = OpenAICompatProvider(
        endpoint,
        model_revision=spec.model_revision,
        cost_ledger=cost_ledger,
        price_per_mtok=price,
        context_note=(f"registry-verified={profile.is_verified()} "
                      f"source={profile.source_url}"),
    )
    provider.model_profile = profile.to_payload()
    provider.probe_artifact_sha256 = spec.probe_artifact_sha256
    return provider


def build_hf_provider(
    spec: Any,
    *,
    role: Literal["agent", "monitor", "fact_checker"],
    device_map: str | dict | None = None,
    seed: int = 0,
    allow_unproven: bool = False,
):
    provider_name = spec.provider
    if provider_name in ENDPOINT_PROVIDERS:
        return build_endpoint_provider(spec, role=role)
    profile = validate_model_assignment(
        spec.model_id,
        role=role,
        quantization=spec.quantization,
        allow_probe_required=bool(spec.probe_artifact_sha256) or allow_unproven,
        hardware_tier=getattr(spec, "hardware_tier", "kaggle_t4"),
    )
    tier = getattr(spec, "hardware_tier", "kaggle_t4")
    if profile.status_for(tier) == "probe_required" and not allow_unproven:
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


def build_provider(
    spec: Any,
    *,
    role: Literal["agent", "monitor", "fact_checker"],
    device_map: str | dict | None = None,
    seed: int = 0,
    allow_unproven: bool = False,
    cost_ledger: Any | None = None,
):
    """Dispatch on the config's provider: transformers or serving plane."""
    if getattr(spec, "provider", "auto") in ENDPOINT_PROVIDERS:
        return build_endpoint_provider(spec, role=role, cost_ledger=cost_ledger)
    return build_hf_provider(
        spec, role=role, device_map=device_map, seed=seed, allow_unproven=allow_unproven
    )


__all__ = ["build_hf_provider", "build_endpoint_provider", "build_provider",
           "ENDPOINT_PROVIDERS"]
