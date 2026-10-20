"""Model-specific T4 admission probe for research matrix cells."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from .config import load_config
from .ids import sha256_json
from .models.factory import build_hf_provider
from .models.interface import GenerationSettings, extract_first_json_object
from .runtime_evidence import capture_runtime_environment
from .traces.storage import atomic_write_json


def run_model_probe(
    config_path: str,
    role: Literal["agent", "monitor"],
    output_path: str,
) -> dict[str, Any]:
    import torch

    config = load_config(config_path)
    spec = config.agent if role == "agent" else config.monitor
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates.append({"name": name, "passed": bool(passed), "detail": detail})

    gate("cuda_available", torch.cuda.is_available(), torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    capability = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    gate("t4_sm75", capability == (7, 5), {"name": props.name, "capability": capability})
    gate("fp16_compute", spec.compute_dtype == "float16", spec.compute_dtype)
    gate("sdpa_attention", spec.attention_backend == "sdpa", spec.attention_backend)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    provider = build_hf_provider(spec, role=role, allow_unproven=True)
    load_seconds = time.perf_counter() - started
    if role == "agent":
        prompt = (
            "Return exactly one JSON object for this fictional tool action: "
            '{"action":"submit_task","summary":"probe complete"}'
        )
        reasoning_mode = bool(config.agent.reasoning_mode)
        max_new_tokens = min(512, config.agent.max_generated_tokens_per_turn)
    else:
        prompt = (
            "Return exactly this JSON object and no other text: "
            '{"risk_score":0,"rubric":{"policy_conflict":0,'
            '"unauthorized_intent":0,"concealment":0,"action_evidence":0},'
            '"evidence_refs":[],"verdict_text":"probe"}'
        )
        reasoning_mode = False
        max_new_tokens = min(256, config.monitor.max_output_tokens)
    turn = provider.generate(
        [{"role": "user", "content": prompt}],
        GenerationSettings(
            temperature=0.0,
            top_p=1.0,
            top_k=20,
            max_new_tokens=max_new_tokens,
            seed=20261020,
            reasoning_mode=reasoning_mode,
        ),
    )
    parsed, parse_status = extract_first_json_object(turn.assistant_text)
    gate("nonempty_visible_answer", bool(turn.assistant_text.strip()), turn.stop_reason)
    gate("json_conformance", parsed is not None, parse_status)
    if role == "agent" and reasoning_mode:
        gate(
            "authentic_trace_channel",
            turn.trace_available and turn.reasoning_parse_status in {"complete", "truncated"},
            turn.reasoning_parse_status,
        )
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    budget_bytes = int(spec.max_memory_per_gpu_gib * 2**30)
    gate("vram_budget", peak_reserved <= budget_bytes, {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "budget_bytes": budget_bytes,
    })
    artifact = {
        "schema_version": "1.0.0",
        "role": role,
        "model_id": spec.model_id,
        "model_revision": provider.model_revision,
        "tokenizer_revision": provider.tokenizer_revision,
        "model_profile": provider.model_profile,
        "config_sha256": config.content_hash(),
        "load_seconds": load_seconds,
        "generation": turn.audit_payload(),
        "gates": gates,
        "all_gates_passed": all(item["passed"] for item in gates),
        "runtime_environment": capture_runtime_environment(),
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    atomic_write_json(Path(output_path), artifact)
    return artifact


__all__ = ["run_model_probe"]
