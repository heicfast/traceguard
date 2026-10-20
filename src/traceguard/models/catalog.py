"""Audited model registry and T4 feasibility policy.

This is a study-design registry, not a leaderboard.  A model is runnable only
when its adapter, reasoning-channel semantics, quantization path, and hardware
probe all pass.  Advertised context length is never treated as a safe runtime
context allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ..ids import sha256_json

Role = Literal["agent", "monitor", "fact_checker"]
T4Status = Literal["validated_path", "probe_required", "blocked"]


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    organization: str
    architecture_family: str
    parameter_billions: float
    native_context_tokens: int
    roles: tuple[Role, ...]
    provider: Literal["qwen_hf", "generic_hf"]
    reasoning_protocol: Literal["qwen3", "think_tags", "none", "harmony_channels"]
    t4_status: T4Status
    t4_runtime: str
    recommended_quantization: Literal["none", "bnb_nf4", "bnb_int8", "native_mxfp4", "gguf"]
    source_url: str
    limitations: str

    @property
    def profile_sha256(self) -> str:
        return sha256_json(asdict(self))

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["profile_sha256"] = self.profile_sha256
        return payload


MODEL_PROFILES: dict[str, ModelProfile] = {
    profile.model_id: profile
    for profile in (
        ModelProfile(
            model_id="Qwen/Qwen3-4B",
            organization="Qwen",
            architecture_family="qwen3_dense",
            parameter_billions=4.0,
            native_context_tokens=32_768,
            roles=("agent", "monitor", "fact_checker"),
            provider="qwen_hf",
            reasoning_protocol="qwen3",
            t4_status="validated_path",
            t4_runtime="fp16 on one T4; SDPA; episode data parallelism",
            recommended_quantization="none",
            source_url="https://huggingface.co/Qwen/Qwen3-4B",
            limitations="Anchor model; same-family agent/monitor results do not establish transfer.",
        ),
        ModelProfile(
            model_id="Qwen/Qwen3-14B",
            organization="Qwen",
            architecture_family="qwen3_dense",
            parameter_billions=14.8,
            native_context_tokens=32_768,
            roles=("agent", "monitor", "fact_checker"),
            provider="qwen_hf",
            reasoning_protocol="qwen3",
            t4_status="probe_required",
            t4_runtime="bitsandbytes NF4 + fp16 compute on one T4; context capped after probe",
            recommended_quantization="bnb_nf4",
            source_url="https://huggingface.co/Qwen/Qwen3-14B",
            limitations="Quantization is a study factor and must not be pooled with fp16 silently.",
        ),
        ModelProfile(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            organization="deepseek-ai",
            architecture_family="qwen2_5_r1_distill",
            parameter_billions=14.8,
            native_context_tokens=131_072,
            roles=("agent", "monitor", "fact_checker"),
            provider="generic_hf",
            reasoning_protocol="think_tags",
            t4_status="probe_required",
            t4_runtime="bitsandbytes NF4 + fp16 compute on one T4; 8K initial cap",
            recommended_quantization="bnb_nf4",
            source_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            limitations="Qwen-derived architecture; organization diversity is not architecture independence.",
        ),
        ModelProfile(
            model_id="ServiceNow-AI/Apriel-1.6-15b-Thinker",
            organization="ServiceNow-AI",
            architecture_family="apriel_thinker",
            parameter_billions=15.0,
            native_context_tokens=32_768,
            roles=("agent", "monitor"),
            provider="generic_hf",
            reasoning_protocol="think_tags",
            t4_status="probe_required",
            t4_runtime="NF4 feasibility and chat-template conformance must pass before admission",
            recommended_quantization="bnb_nf4",
            source_url="https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker",
            limitations="Parser and tool-use conformance are unvalidated; exploratory candidate only.",
        ),
        ModelProfile(
            model_id="google/gemma-4-12B-it",
            organization="google",
            architecture_family="gemma4",
            parameter_billions=12.0,
            native_context_tokens=131_072,
            roles=("monitor", "fact_checker"),
            provider="generic_hf",
            reasoning_protocol="none",
            t4_status="probe_required",
            t4_runtime="NF4 text-only monitor path; gated weights and template probe required",
            recommended_quantization="bnb_nf4",
            source_url="https://huggingface.co/google/gemma-4-12B-it",
            limitations="No admissible raw reasoning channel for the primary agent treatment.",
        ),
        ModelProfile(
            model_id="openai/gpt-oss-20b",
            organization="openai",
            architecture_family="gpt_oss_moe",
            parameter_billions=20.9,
            native_context_tokens=131_072,
            roles=("monitor", "fact_checker"),
            provider="generic_hf",
            reasoning_protocol="harmony_channels",
            t4_status="blocked",
            t4_runtime="official MXFP4 requires Hopper-or-newer; bf16 path is about 48 GB",
            recommended_quantization="native_mxfp4",
            source_url="https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers",
            limitations="Not a canonical T4 arm; third-party GGUF would be a separate backend treatment.",
        ),
        ModelProfile(
            model_id="Qwen/Qwen3.8-27B",
            organization="Qwen",
            architecture_family="qwen3_8_dense",
            parameter_billions=27.0,
            native_context_tokens=262_144,
            roles=("agent", "monitor"),
            provider="generic_hf",
            reasoning_protocol="think_tags",
            t4_status="blocked",
            t4_runtime="requires a separately validated low-bit backend; official FP8 is not a T4 path",
            recommended_quantization="gguf",
            source_url="https://huggingface.co/Qwen/Qwen3.8-27B",
            limitations="A 3-bit community quantization is not equivalent to the official checkpoint.",
        ),
    )
}


def get_model_profile(model_id: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[model_id]
    except KeyError as exc:
        raise ValueError(
            f"model {model_id!r} is not in the audited registry; add a sourced profile and "
            "hardware probe before research use"
        ) from exc


def validate_model_assignment(
    model_id: str,
    *,
    role: Role,
    quantization: str,
    allow_probe_required: bool,
) -> ModelProfile:
    profile = get_model_profile(model_id)
    if role not in profile.roles:
        raise ValueError(f"{model_id} is not approved for role {role}")
    if profile.t4_status == "blocked":
        raise ValueError(f"{model_id} is blocked on T4: {profile.t4_runtime}")
    if profile.t4_status == "probe_required" and not allow_probe_required:
        raise ValueError(f"{model_id} requires a recorded hardware/conformance probe")
    if quantization != profile.recommended_quantization:
        raise ValueError(
            f"{model_id} requires declared quantization {profile.recommended_quantization!r}; "
            f"got {quantization!r}"
        )
    return profile


__all__ = [
    "MODEL_PROFILES",
    "ModelProfile",
    "get_model_profile",
    "validate_model_assignment",
]
