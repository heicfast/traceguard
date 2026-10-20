"""Regression tests for the audited multi-model architecture."""

from __future__ import annotations

import pytest

from traceguard.config import AgentConfig, load_config
from traceguard.model_matrix import ModelMatrix, build_schedule, materialize_matrix
from traceguard.models.catalog import get_model_profile, validate_model_assignment


def _model(model_id: str, *, provider: str, protocol: str, quantization: str,
           probe: str | None = None) -> dict:
    return {
        "model_id": model_id,
        "revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "provider": provider,
        "reasoning_protocol": protocol,
        "quantization": quantization,
        "compute_dtype": "float16",
        "attention_backend": "sdpa",
        "max_context_tokens": 8192,
        "probe_artifact_sha256": probe,
    }


def test_t4_registry_blocks_mxfp4_gpt_oss():
    profile = get_model_profile("openai/gpt-oss-20b")
    assert profile.t4_status == "blocked"
    with pytest.raises(ValueError, match="blocked on T4"):
        validate_model_assignment(
            profile.model_id,
            role="monitor",
            quantization="native_mxfp4",
            allow_probe_required=True,
        )


def test_probe_required_model_cannot_enter_matrix_without_artifact():
    with pytest.raises(ValueError, match="recorded hardware/conformance probe"):
        validate_model_assignment(
            "Qwen/Qwen3-14B",
            role="agent",
            quantization="bnb_nf4",
            allow_probe_required=False,
        )


def test_reasoning_mode_cannot_use_no_trace_protocol():
    with pytest.raises(ValueError, match="trace-producing"):
        AgentConfig(reasoning_mode=True, reasoning_protocol="none")


def test_matrix_deduplicates_generation_cohorts(tmp_path):
    qwen4 = _model(
        "Qwen/Qwen3-4B", provider="qwen_hf", protocol="qwen3", quantization="none"
    )
    qwen14 = _model(
        "Qwen/Qwen3-14B",
        provider="qwen_hf",
        protocol="qwen3",
        quantization="bnb_nf4",
        probe="c" * 64,
    )
    matrix = ModelMatrix.model_validate({
        "study_id": "test",
        "benchmark_config": "configs/phase3_pilot.yaml",
        "frozen_code_revision": "d" * 40,
        "cells": [
            {
                "cell_id": "anchor",
                "generation_cohort_id": "agent-qwen4",
                "agent": qwen4,
                "monitor": qwen4,
                "account_slot": 0,
                "replicate_ids": [0, 1],
                "primary": True,
            },
            {
                "cell_id": "scaled-monitor",
                "generation_cohort_id": "agent-qwen4",
                "agent": qwen4,
                "monitor": qwen14,
                "account_slot": 0,
                "replicate_ids": [0, 1],
                "primary": False,
            },
        ],
    })
    schedule = build_schedule(matrix)
    assert len(schedule["generation_jobs"]) == 2
    assert len(schedule["monitoring_jobs"]) == 4
    assert {
        job["generation_job_id"] for job in schedule["monitoring_jobs"]
    } == {"agent-qwen4.r0", "agent-qwen4.r1"}
    materialized = materialize_matrix(matrix, tmp_path / "matrix")
    all_jobs = materialized["generation_jobs"] + materialized["monitoring_jobs"]
    configs = [load_config(job["config_path"]) for job in all_jobs]
    assert len(configs) == 6
    assert {cfg.provenance.replicate_id for cfg in configs} == {0, 1}
