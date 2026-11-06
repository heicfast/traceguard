"""Audited model registry and hardware-admission policy (v0.7.0).

This is a study-design registry, not a leaderboard.  A model is runnable only
when its adapter, reasoning-channel semantics, quantization path, and a
hardware probe on the TARGET TIER all pass.  Advertised context length is
never treated as a safe runtime context allocation.

v0.7.0 changes (docs/UPGRADE_v0.7.md):

* hardware tiers replace the T4-only feasibility field — ``kaggle_t4`` (the
  validated legacy path), ``aws_a100_80gb`` (p4de/g6e-class), and
  ``aws_h100_80gb`` (p5-class).  Every tier has its OWN admission status;
  nothing is assumed to transfer across hardware;
* entries carry a ``verification`` record — ``verified`` (source opened and
  checked on the stated date) or ``candidate`` (existence/ID reported by a
  credible source but not yet confirmed by us; the admission probe fails
  closed until the operator confirms the exact repo/API id);
* API frontier models (no local weights) are registered with their
  OpenAI-compatible endpoints and pricing sources; they are MONITOR-role
  only in the default design (agents remain open-weights so the full
  evidence contract, including token ids, applies to the behavior under
  study).

Sources for every verified entry are in docs/REFERENCES.md (§Model and
pricing sources, checked 2026-11-06).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ..ids import sha256_json

Role = Literal["agent", "monitor", "fact_checker"]
HardwareTier = Literal["kaggle_t4", "aws_a100_80gb", "aws_h100_80gb"]
HardwareStatus = Literal["validated_path", "probe_required", "blocked"]
ModelKind = Literal["local_weights", "remote_api"]


@dataclass(frozen=True)
class Verification:
    status: Literal["verified", "candidate"]
    verified_at: str  # ISO date of OUR source check
    source_url: str

    def validate(self) -> None:
        if self.status not in ("verified", "candidate"):
            raise ValueError("verification status must be verified|candidate")
        if not self.source_url.startswith("http"):
            raise ValueError("verification must cite a source URL")


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    organization: str
    architecture_family: str
    parameter_billions: float
    native_context_tokens: int
    roles: tuple[Role, ...]
    kind: ModelKind = "local_weights"
    provider: Literal["qwen_hf", "generic_hf", "vllm_openai", "api_openai"] = "generic_hf"
    reasoning_protocol: Literal["qwen3", "think_tags", "none", "harmony_channels",
                                 "reasoning_field"] = "none"
    # Per-tier admission: nothing transfers across hardware without a probe.
    hardware_status: dict[str, HardwareStatus] = field(default_factory=dict)
    hardware_notes: dict[str, str] = field(default_factory=dict)
    recommended_quantization: str = "none"
    # vLLM serving defaults for the AWS tiers (see serving.ServerSpec).
    serving_quantization: str | None = None
    serving_dtype: str = "bfloat16"
    serving_tensor_parallel: int = 1
    serving_reasoning_parser: str | None = None
    source_url: str = ""
    limitations: str = ""
    verification: Verification | None = None

    @property
    def profile_sha256(self) -> str:
        return sha256_json(asdict(self))

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["profile_sha256"] = self.profile_sha256
        return payload

    def status_for(self, tier: str) -> HardwareStatus:
        return self.hardware_status.get(tier, "blocked")

    def is_verified(self) -> bool:
        return self.verification is not None and self.verification.status == "verified"


def _v(status: str, date: str, url: str) -> Verification:
    entry = Verification(status=status, verified_at=date, source_url=url)  # type: ignore[arg-type]
    entry.validate()
    return entry


T4_KAGGLE_NOTE = "validated on Kaggle T4 during the v0.5/v0.6 program"
CHECKED = "2026-11-06"

MODEL_PROFILES: dict[str, ModelProfile] = {
    profile.model_id: profile
    for profile in (
        # ---- local open-weights: the agent/monitor workhorses -----------------
        ModelProfile(
            model_id="Qwen/Qwen3-4B",
            organization="Qwen",
            architecture_family="qwen3_dense",
            parameter_billions=4.0,
            native_context_tokens=32_768,
            roles=("agent", "monitor", "fact_checker"),
            kind="local_weights",
            provider="qwen_hf",
            reasoning_protocol="qwen3",
            hardware_status={"kaggle_t4": "validated_path",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "fp16 on one T4; SDPA (T4 program)",
                             "aws_a100_80gb": "bf16, fits one GPU with large KV headroom",
                             "aws_h100_80gb": "bf16; FA3-class kernels via vLLM V1"},
            recommended_quantization="none",
            serving_dtype="bfloat16",
            serving_reasoning_parser="qwen3",
            source_url="https://huggingface.co/Qwen/Qwen3-4B",
            limitations="Anchor model; same-family agent/monitor results do not establish transfer.",
            verification=_v("verified", "2026-10-05",
                             "https://huggingface.co/Qwen/Qwen3-4B"),
        ),
        ModelProfile(
            model_id="Qwen/Qwen3-14B",
            organization="Qwen",
            architecture_family="qwen3_dense",
            parameter_billions=14.8,
            native_context_tokens=32_768,
            roles=("agent", "monitor", "fact_checker"),
            kind="local_weights",
            provider="qwen_hf",
            reasoning_protocol="qwen3",
            hardware_status={"kaggle_t4": "probe_required",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "bitsandbytes NF4 + fp16 compute (v6 program)",
                             "aws_a100_80gb": "bf16 (~30 GiB) on one A100-80; quantization-free arm",
                             "aws_h100_80gb": "bf16 on one H100; FA3-class kernels"},
            recommended_quantization="bnb_nf4",
            serving_dtype="bfloat16",
            serving_reasoning_parser="qwen3",
            source_url="https://huggingface.co/Qwen/Qwen3-14B",
            limitations="Quantization is a study factor and must not be pooled with bf16 silently.",
            verification=_v("verified", "2026-10-05",
                             "https://huggingface.co/Qwen/Qwen3-14B"),
        ),
        ModelProfile(
            model_id="Qwen/Qwen3-32B",
            organization="Qwen",
            architecture_family="qwen3_dense",
            parameter_billions=32.8,
            native_context_tokens=32_768,
            roles=("agent", "monitor", "fact_checker"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="qwen3",
            hardware_status={"kaggle_t4": "blocked",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "does not fit the T4 program",
                             "aws_a100_80gb": "bf16 ~66 GiB: tp=2 on A100-80, or tp=1 with the official FP8 checkpoint",
                             "aws_h100_80gb": "bf16 tp=2 or FP8 tp=1; FA3-class kernels"},
            recommended_quantization="none",
            serving_quantization="fp8",
            serving_dtype="bfloat16",
            serving_tensor_parallel=2,
            serving_reasoning_parser="qwen3",
            source_url="https://huggingface.co/Qwen/Qwen3-32B",
            limitations="New monitor-scale arm; probe must confirm revision, parser, and VRAM.",
            verification=_v("verified", CHECKED, "https://huggingface.co/Qwen/Qwen3-32B"),
        ),
        ModelProfile(
            model_id="Qwen/Qwen3.5-14B",
            organization="Qwen",
            architecture_family="qwen3_5",
            parameter_billions=14.0,
            native_context_tokens=262_144,
            roles=("agent", "monitor", "fact_checker"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="qwen3",
            hardware_status={"kaggle_t4": "blocked",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "out of T4 scope",
                             "aws_a100_80gb": "bf16 on one A100-80 (probe required)",
                             "aws_h100_80gb": "bf16 on one H100"},
            recommended_quantization="none",
            serving_dtype="bfloat16",
            serving_reasoning_parser="qwen3",
            source_url="https://huggingface.co/Qwen",
            limitations=(
                "CANDIDATE: the Qwen3.5 open-weight series (0.5B-32B) was announced "
                "2026-02-15 (qwen.ai/blog?id=qwen3.5, checked " + CHECKED + "); the exact "
                "HF repo id, template, and parser behavior must be confirmed by the "
                "admission probe before any research use."
            ),
            verification=_v("candidate", CHECKED, "https://qwen.ai/blog?id=qwen3.5"),
        ),
        ModelProfile(
            model_id="Qwen/Qwen3.8-27B",
            organization="Qwen",
            architecture_family="qwen3_8_dense",
            parameter_billions=27.0,
            native_context_tokens=262_144,
            roles=("agent", "monitor"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="think_tags",
            hardware_status={"kaggle_t4": "blocked",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "official FP8 is not a T4 path (v0.6 registry)",
                             "aws_a100_80gb": "FP8 tp=1 or bf16 tp=2 (probe required)",
                             "aws_h100_80gb": "FP8 tp=1; FA3-class kernels"},
            recommended_quantization="gguf",
            serving_quantization="fp8",
            serving_dtype="bfloat16",
            serving_tensor_parallel=2,
            serving_reasoning_parser="qwen3",
            source_url="https://huggingface.co/Qwen/Qwen3.8-27B",
            limitations="Third-party low-bit quants are not equivalent to the official checkpoint.",
            verification=_v("candidate", "2026-10-05",
                             "https://huggingface.co/Qwen/Qwen3.8-27B"),
        ),
        ModelProfile(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            organization="deepseek-ai",
            architecture_family="qwen2_5_r1_distill",
            parameter_billions=14.8,
            native_context_tokens=131_072,
            roles=("agent", "monitor", "fact_checker"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="think_tags",
            hardware_status={"kaggle_t4": "probe_required",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "bitsandbytes NF4 + fp16 (v6 program)",
                             "aws_a100_80gb": "bf16 on one A100-80; server reasoning parser deepseek_r1",
                             "aws_h100_80gb": "bf16 on one H100"},
            recommended_quantization="bnb_nf4",
            serving_dtype="bfloat16",
            serving_reasoning_parser="deepseek_r1",
            source_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            limitations="Qwen-derived architecture; organization diversity is not architecture independence.",
            verification=_v("verified", "2026-10-05",
                             "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
        ),
        ModelProfile(
            model_id="openai/gpt-oss-20b",
            organization="openai",
            architecture_family="gpt_oss_moe",
            parameter_billions=20.9,
            native_context_tokens=131_072,
            roles=("monitor", "fact_checker"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="harmony_channels",
            hardware_status={"kaggle_t4": "blocked",
                              "aws_a100_80gb": "blocked",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={
                "kaggle_t4": "MXFP4 requires Hopper-or-newer NVIDIA or MI300x-class AMD (vLLM gpt-oss announcement, verified "
                             + CHECKED + ") — blocked on Turing and on Ampere",
                "aws_a100_80gb": "BLOCKED: Ampere (sm80) is not an MXFP4 target; use the p5 tier",
                "aws_h100_80gb": "native MXFP4 on Hopper via vLLM (supported since v0.11, verified "
                                  + CHECKED + ")",
            },
            recommended_quantization="native_mxfp4",
            serving_quantization="mxfp4",
            serving_dtype="bfloat16",
            serving_reasoning_parser=None,
            source_url="https://huggingface.co/openai/gpt-oss-20b",
            limitations="Harmony format; monitor-role only in the default design.",
            verification=_v("verified", CHECKED, "https://vllm.ai/blog/2025-08-05-gpt-oss"),
        ),
        ModelProfile(
            model_id="zai-org/GLM-4.5-Air-FP8",
            organization="zai-org",
            architecture_family="glm_4_5_moe",
            parameter_billions=106.0,
            native_context_tokens=131_072,
            roles=("monitor", "fact_checker"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="think_tags",
            hardware_status={"kaggle_t4": "blocked",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "out of T4 scope",
                             "aws_a100_80gb": "official FP8 weights ~106 GiB: tp=8 on one p4de node",
                             "aws_h100_80gb": "tp=8 on one p5 node; FA3-class kernels"},
            recommended_quantization="fp8",
            serving_quantization="fp8",
            serving_dtype="bfloat16",
            serving_tensor_parallel=8,
            serving_reasoning_parser="glm45",
            source_url="https://huggingface.co/zai-org/GLM-4.5-Air-FP8",
            limitations=(
                "CANDIDATE: cross-family MoE monitor arm; the admission probe must confirm "
                "the FP8 repo id, template, and single-node VRAM fit before research use."
            ),
            verification=_v("candidate", CHECKED,
                             "https://huggingface.co/zai-org/GLM-4.5-Air-FP8"),
        ),
        ModelProfile(
            model_id="ServiceNow-AI/Apriel-1.6-15b-Thinker",
            organization="ServiceNow-AI",
            architecture_family="apriel_thinker",
            parameter_billions=15.0,
            native_context_tokens=32_768,
            roles=("agent", "monitor"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="think_tags",
            hardware_status={"kaggle_t4": "probe_required",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "NF4 feasibility and chat-template conformance (v0.6)",
                             "aws_a100_80gb": "bf16 on one A100-80 (probe required)",
                             "aws_h100_80gb": "bf16 on one H100 (probe required)"},
            recommended_quantization="bnb_nf4",
            serving_dtype="bfloat16",
            serving_reasoning_parser="deepseek_r1",
            source_url="https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker",
            limitations="Parser and tool-use conformance are unvalidated; exploratory candidate only.",
            verification=_v("candidate", "2026-10-05",
                             "https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker"),
        ),
        ModelProfile(
            model_id="google/gemma-4-12B-it",
            organization="google",
            architecture_family="gemma4",
            parameter_billions=12.0,
            native_context_tokens=131_072,
            roles=("monitor", "fact_checker"),
            kind="local_weights",
            provider="vllm_openai",
            reasoning_protocol="none",
            hardware_status={"kaggle_t4": "probe_required",
                              "aws_a100_80gb": "probe_required",
                              "aws_h100_80gb": "probe_required"},
            hardware_notes={"kaggle_t4": "NF4 text-only monitor path (v0.6)",
                             "aws_a100_80gb": "bf16 on one A100-80; gated weights",
                             "aws_h100_80gb": "bf16 on one H100; gated weights"},
            recommended_quantization="bnb_nf4",
            serving_dtype="bfloat16",
            source_url="https://huggingface.co/google/gemma-4-12B-it",
            limitations="No admissible raw reasoning channel for the primary agent treatment.",
            verification=_v("candidate", "2026-10-05",
                             "https://huggingface.co/google/gemma-4-12B-it"),
        ),
        # ---- remote API frontier monitors (no local weights) ------------------
        ModelProfile(
            model_id="api/openai/gpt-6.1-sol",
            organization="openai",
            architecture_family="gpt6_api",
            parameter_billions=0.0,
            native_context_tokens=1_050_000,
            roles=("monitor", "fact_checker"),
            kind="remote_api",
            provider="api_openai",
            reasoning_protocol="reasoning_field",
            hardware_status={},
            hardware_notes={"remote": "OpenAI chat completions endpoint; reasoning summary only"},
            recommended_quantization="none",
            source_url="https://developers.openai.com/api/docs/pricing",
            limitations=(
                "Frontier API monitor arm. Exact API model ids drift; the admission probe "
                "must LIST /v1/models and pin the served id + date. Determinism class: "
                "unseeded_api. Pricing verified " + CHECKED + " ($2/$10 per 1M tok)."
            ),
            verification=_v("verified", CHECKED,
                             "https://developers.openai.com/api/docs/pricing"),
        ),
        ModelProfile(
            model_id="api/openai/gpt-6-luna",
            organization="openai",
            architecture_family="gpt6_api",
            parameter_billions=0.0,
            native_context_tokens=1_050_000,
            roles=("monitor", "fact_checker"),
            kind="remote_api",
            provider="api_openai",
            reasoning_protocol="reasoning_field",
            hardware_status={},
            hardware_notes={"remote": "high-volume tier; $0.10/$0.50 per 1M tok (verified "
                                        + CHECKED + ")"},
            recommended_quantization="none",
            source_url="https://developers.openai.com/api/docs/pricing",
            limitations="API monitor arm; probe pins the served model id.",
            verification=_v("verified", CHECKED,
                             "https://developers.openai.com/api/docs/pricing"),
        ),
        ModelProfile(
            model_id="api/anthropic/claude-opus-4-6",
            organization="anthropic",
            architecture_family="claude_api",
            parameter_billions=0.0,
            native_context_tokens=1_000_000,
            roles=("monitor", "fact_checker"),
            kind="remote_api",
            provider="api_openai",
            reasoning_protocol="reasoning_field",
            hardware_status={},
            hardware_notes={"remote": "Claude Opus 4.6 announced 2026 (anthropic.com/news/claude-opus-4-6, "
                                        "checked " + CHECKED + "); endpoint and exact served id "
                                        "pinned by the admission probe"},
            recommended_quantization="none",
            source_url="https://www.anthropic.com/news/claude-opus-4-6",
            limitations=(
                "CANDIDATE API id: the model is verified to exist; the exact API model "
                "string must be confirmed against the provider's model list at probe time."
            ),
            verification=_v("candidate", CHECKED,
                             "https://www.anthropic.com/news/claude-opus-4-6"),
        ),
        ModelProfile(
            model_id="api/deepseek/deepseek-chat",
            organization="deepseek",
            architecture_family="deepseek_api",
            parameter_billions=0.0,
            native_context_tokens=131_072,
            roles=("monitor", "fact_checker"),
            kind="remote_api",
            provider="api_openai",
            reasoning_protocol="reasoning_field",
            hardware_status={},
            hardware_notes={"remote": "api-docs.deepseek.com; deepseek-chat currently routes to "
                                        "the V4.x generation (verified " + CHECKED + ")"},
            recommended_quantization="none",
            source_url="https://api-docs.deepseek.com/",
            limitations="Served generation drifts; probe pins the underlying version via the models endpoint.",
            verification=_v("verified", CHECKED, "https://api-docs.deepseek.com/"),
        ),
    )
}

# Keep the legacy attribute working for older call sites / tests.
T4Status = HardwareStatus


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
    hardware_tier: str = "kaggle_t4",
) -> ModelProfile:
    profile = get_model_profile(model_id)
    if role not in profile.roles:
        raise ValueError(f"{model_id} is not approved for role {role}")
    if profile.kind == "remote_api":
        if quantization not in ("none", ""):
            raise ValueError(f"{model_id} is a remote API model; quantization is not applicable")
        if not allow_probe_required:
            raise ValueError(f"{model_id} requires its admission probe (served-id pinning)")
        return profile
    status = profile.status_for(hardware_tier)
    if status == "blocked":
        raise ValueError(
            f"{model_id} is blocked on {hardware_tier}: "
            f"{profile.hardware_notes.get(hardware_tier, 'no admissible path')}"
        )
    if status == "probe_required" and not allow_probe_required:
        raise ValueError(
            f"{model_id} requires a recorded hardware/conformance probe on {hardware_tier}"
        )
    if hardware_tier in ("aws_a100_80gb", "aws_h100_80gb"):
        # Serving-tier quantization is a ServerSpec concern, checked there;
        # the legacy quantization field must simply be 'none' (server path)
        # or the legacy bnb recipe (transformers path).
        allowed = {"none", "bnb_nf4", "bnb_int8"}
        if quantization not in allowed:
            raise ValueError(
                f"{model_id} on {hardware_tier}: quantization must be one of "
                f"{sorted(allowed)}; server-side quantization belongs to the serving spec"
            )
    elif quantization != profile.recommended_quantization:
        raise ValueError(
            f"{model_id} requires declared quantization {profile.recommended_quantization!r}; "
            f"got {quantization!r}"
        )
    return profile


__all__ = [
    "MODEL_PROFILES",
    "ModelProfile",
    "Verification",
    "HardwareTier",
    "get_model_profile",
    "validate_model_assignment",
    "T4Status",
]
