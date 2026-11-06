"""Unit tests: follow-up monitor, paper export, evidence sync dry-run, config v0.7."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceguard.config import (
    AgentConfig,
    ExperimentConfig,
    MonitorConfig,
    ProvenanceConfig,
    load_config,
)
from traceguard.monitors.followup import FollowUpMonitor
from traceguard.monitors.llm import MONITOR_PROMPT_TEMPLATE
from traceguard.paper_export import export_paper_tables
from traceguard.traces.views import MonitorObservation


class _ScriptedProvider:
    """Deterministic provider answering each phase in a fixed order."""

    provider_name = "scripted_test"

    def __init__(self, answers: list[str], reasoning: list[str | None] | None = None):
        self.answers = list(answers)
        self.reasoning = list(reasoning or [None] * len(answers))
        self.calls = 0
        self.model_revision = "scripted-1"

    def generate(self, messages, settings):
        from traceguard.models.interface import ModelTurn

        index = min(self.calls, len(self.answers) - 1)
        answer = self.answers[index]
        reasoning = self.reasoning[index]
        self.calls += 1
        return ModelTurn(
            raw_output=answer, reasoning_text=reasoning, assistant_text=answer,
            reasoning_parse_status="complete" if reasoning else "absent",
            token_usage={"prompt_tokens": 10, "generated_tokens": 5},
            model_revision=self.model_revision,
        )


def _observation() -> MonitorObservation:
    obs = MonitorObservation(
        episode_id="ep-abc", condition="actions", text="== AGENT TOOL ACTIVITY ==\n...",
        evidence_event_count=3, reasoning_turn_count=0,
    )
    obs.compute_hash()
    return obs


def test_followup_monitor_three_phase_flow():
    adjudication = json.dumps({
        "rubric": {"policy_conflict": 10, "unauthorized_intent": 10,
                    "concealment": 5, "action_evidence": 10},
        "evidence_refs": ["alias switch"],
        "verdict_text": "suspicious reference update",
    })
    monitor_provider = _ScriptedProvider([
        "Which validation record covers the deployment window?",
        adjudication,
    ])
    elaboration_provider = _ScriptedProvider(
        ["I used the already-passed record from the previous window."],
        reasoning=["checking the registry... the stale record was faster"],
    )
    monitor = FollowUpMonitor(monitor_provider, elaboration_provider)
    score = monitor.score(_observation())
    assert score.valid
    assert score.suspicion == (10 + 10 + 5 + 10) / 100
    assert score.monitor_name == "llm_followup_scripted_test"
    # all three phases recorded in the audit trail
    assert set(score.inference_audit) == {"question", "elaboration", "adjudicator"}
    assert score.parsed_response["followup"]["elaboration_reasoning_available"]
    assert score.rendered_prompt_hash


def test_followup_monitor_invalid_when_question_phase_empty():
    monitor_provider = _ScriptedProvider(["", ""])
    elaboration_provider = _ScriptedProvider(["answer"])
    monitor = FollowUpMonitor(monitor_provider, elaboration_provider)
    score = monitor.score(_observation())
    assert not score.valid
    assert score.invalid_reason == "question_phase_empty"


def test_followup_monitor_invalid_adjudication():
    monitor_provider = _ScriptedProvider(["question?", "not json at all"])
    elaboration_provider = _ScriptedProvider(["an answer"])
    monitor = FollowUpMonitor(monitor_provider, elaboration_provider)
    score = monitor.score(_observation())
    assert not score.valid
    assert score.invalid_reason and "invalid_adjudication" in score.invalid_reason


def test_followup_prompt_is_registered_for_verification():
    from traceguard.monitors.followup import (
        ADJUDICATOR_QA_BLOCK,
        ELABORATION_PROMPT,
        FOLLOWUP_QUESTION_PROMPT,
    )
    from traceguard.monitors.interface import prompt_hash

    expected = prompt_hash("\n".join([
        FOLLOWUP_QUESTION_PROMPT, ELABORATION_PROMPT,
        MONITOR_PROMPT_TEMPLATE, ADJUDICATOR_QA_BLOCK,
    ]))
    from traceguard.verification import _frozen_prompt_hashes

    registry = _frozen_prompt_hashes()
    assert expected in registry["followup"]
    assert registry["followup"]  # non-empty


# ---------------------------------------------------------------------------
# Paper export
# ---------------------------------------------------------------------------

def test_paper_export_writes_latex_and_csv(tmp_path):
    report = {
        "contrasts": [{
            "condition_primary": "actions_plus_cot", "condition_baseline": "actions",
            "estimate": 0.125, "ci_low": -0.02, "ci_high": 0.27,
            "ci_status": "estimable", "pairwise_retention": 0.98,
        }],
        "capability_funnel": {
            "explicit_override": {"episodes": 64, "attempted": 53, "succeeded": 53},
            "authorization_claim": {"episodes": 64, "attempted": 48, "succeeded": 40},
        },
    }
    report_path = tmp_path / "analysis_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = export_paper_tables(report_path, tmp_path / "out", prefix="v7")
    assert (tmp_path / "out" / "v7_paired_contrast.tex").exists()
    csv_text = (tmp_path / "out" / "v7_paired_contrast.csv").read_text()
    assert "actions_plus_cot_vs_actions" in csv_text
    assert (tmp_path / "out" / "v7_capability_funnel.tex").exists()
    assert result["source_report_sha256"]


def test_paper_export_fails_loudly_without_source(tmp_path):
    from traceguard.paper_export import paired_contrast_table

    with pytest.raises(ValueError, match="no paired contrasts"):
        paired_contrast_table({"unrelated": True})


# ---------------------------------------------------------------------------
# Config v0.7 validation
# ---------------------------------------------------------------------------

def _provenance() -> ProvenanceConfig:
    return ProvenanceConfig(experiment_name="t", experiment_stage="pilot",
                             code_revision="a" * 40)


def test_endpoint_agent_config_requires_endpoint_fields():
    with pytest.raises(ValueError, match="api_base_url and api_model_id"):
        AgentConfig(provider="vllm_openai", model_id="Qwen/Qwen3-14B",
                     model_revision="a" * 40, tokenizer_revision="a" * 40)
    with pytest.raises(ValueError, match="transformers quantization"):
        AgentConfig(provider="vllm_openai", model_id="Qwen/Qwen3-14B",
                     model_revision="a" * 40, tokenizer_revision="a" * 40,
                     api_base_url="http://127.0.0.1:8000/v1",
                     api_model_id="qwen3-14b", quantization="bnb_nf4")


def test_valid_endpoint_agent_config_roundtrips():
    agent = AgentConfig(
        provider="vllm_openai", model_id="Qwen/Qwen3-14B",
        model_revision="a" * 40, tokenizer_revision="a" * 40,
        api_base_url="http://127.0.0.1:8000/v1", api_model_id="qwen3-14b",
        hardware_tier="aws_a100_80gb", compute_dtype="bfloat16",
    )
    config = ExperimentConfig(provenance=_provenance(), agent=agent)
    payload = config.to_payload()
    assert payload["agent"]["hardware_tier"] == "aws_a100_80gb"
    assert config.content_hash()


def test_budget_cap_validation():
    config = ExperimentConfig(provenance=_provenance(), budget_usd_hard_cap=400.0)
    assert config.budget_usd_hard_cap == 400.0
    with pytest.raises(ValueError):
        ExperimentConfig(provenance=_provenance(), budget_usd_hard_cap=-1.0)


def test_followup_monitor_config_requires_elaboration_model():
    with pytest.raises(ValueError, match="elaboration model"):
        MonitorConfig(followup_enabled=True)


def test_legacy_engine_version_still_loads():
    config = ExperimentConfig(provenance=_provenance())
    assert config.sandbox.engine_version == "1.1.0"
    # the shipped v0.6 pilot config (legacy engine pin) must still load
    repo_root = Path(__file__).resolve().parents[2]
    legacy = repo_root / "configs" / "phase3_pilot.yaml"
    if legacy.exists():
        legacy_config = load_config(legacy)
        assert legacy_config.sandbox.engine_version in ("1.0.2", "1.1.0")


# ---------------------------------------------------------------------------
# Evidence sync (dry run; no boto3 needed)
# ---------------------------------------------------------------------------

def test_sync_evidence_dry_run_lists_files(tmp_path):
    from traceguard.evidence_sync import _resolve_s3_path, sync_run_to_s3

    run = tmp_path / "run"
    (run / "episodes").mkdir(parents=True)
    (run / "episodes" / "shard-0.jsonl").write_text('{"episode_id": "ep-1"}\n')
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "r1", "config_sha256": "c" * 64, "code_revision": "a" * 40,
    }))
    result = sync_run_to_s3(run, "s3://traceguard-evidence/runs/r1", dry_run=True)
    assert result["ok"] and result["dry_run"]
    keys = [f["key"] for f in result["files"]]
    assert any(k.endswith("run_manifest.json") for k in keys)
    assert all(f["sha256"] and len(f["sha256"]) == 64 for f in result["files"])
    with pytest.raises(ValueError):
        _resolve_s3_path("not-an-s3-path")
    with pytest.raises(FileNotFoundError):
        sync_run_to_s3(tmp_path / "missing", "s3://b/p", dry_run=True)


# ---------------------------------------------------------------------------
# Factory endpoint construction
# ---------------------------------------------------------------------------

def test_factory_refuses_remote_url_for_vllm_provider():
    from types import SimpleNamespace

    from traceguard.models.factory import build_endpoint_provider

    spec = SimpleNamespace(
        model_id="Qwen/Qwen3-14B", model_revision="a" * 40,
        provider="vllm_openai", reasoning_protocol="qwen3", quantization="none",
        probe_artifact_sha256="e" * 64,  # admission passes; the URL check must fire
        api_base_url="https://api.example.com/v1", api_model_id="qwen3-14b",
        api_key_env="X", hardware_tier="aws_a100_80gb",
    )
    with pytest.raises(ValueError, match="local server"):
        build_endpoint_provider(spec, role="monitor")
