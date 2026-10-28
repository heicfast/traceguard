"""Regression tests for the v2 raw-evidence and transactional-storage contract."""

from __future__ import annotations

import json

from traceguard.ids import sha256_json, sha256_text
from traceguard.models.interface import ModelTurn
from traceguard.traces.storage import RunStore, atomic_write_json


def test_model_turn_audit_payload_is_lossless_and_hash_linked():
    turn = ModelTurn(
        raw_output="reasoning\nanswer",
        raw_output_with_special_tokens="<think>reasoning</think>\nanswer<|im_end|>",
        reasoning_text="reasoning",
        assistant_text="answer",
        generated_token_ids=[1, 2, 151668, 3],
        prompt_token_ids=[91, 92],
        rendered_prompt="system\nuser",
        input_messages=[{"role": "user", "content": "user"}],
        generation_parameters={"temperature": 0.6, "seed": 7},
        runtime_metadata={"torch": "test"},
    )
    audit = turn.audit_payload()
    assert audit["raw_output"] == turn.raw_output
    assert audit["raw_output_with_special_tokens"] == turn.raw_output_with_special_tokens
    assert audit["generated_token_ids"] == [1, 2, 151668, 3]
    assert audit["raw_output_utf8_sha256"] == sha256_text(turn.raw_output)
    assert audit["generated_token_ids_sha256"] == sha256_json([1, 2, 151668, 3])
    assert audit["rendered_prompt_utf8_sha256"] == sha256_text("system\nuser")


def test_episode_transaction_recovers_without_regeneration(tmp_path):
    """Recovery projects a schema-VALID, plan-member transaction (security
    review F1: validation + membership checks are part of the contract)."""
    from pathlib import Path

    from traceguard.config import load_config
    from traceguard.models.fake import TaskScriptedAgent
    from traceguard.monitors.rules import RuleMonitor
    from traceguard.online_runtime import _assemble_records
    from traceguard.protocols.pre_action import run_online_episode
    from traceguard.tasks.generator import build_episode_plan

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml")
    plan = next(p for p in build_episode_plan(cfg, stage="smoke")
                if p.policy_mode == "attack")
    task = plan.load_task_instance()
    provider = TaskScriptedAgent(task, "attack")
    outcome = run_online_episode(
        plan=plan, task=task, provider=provider, monitor=RuleMonitor(),
        config=cfg, run_id="recover-it", threshold=1.5)

    class _P:
        provider_name = "test"
        model_revision = "test-rev"
        model_profile = None

    record = _assemble_records(
        outcome_payload=outcome.payload, plan=plan, task=task,
        config=cfg, provider=_P(), run_id="recover-it", wall_ms=1.0)
    envelope, privileged = record.envelope, record.privileged
    prov = envelope["provenance"]

    store = RunStore(tmp_path / "run")
    store.write_plan_manifest({
        "schema_version": "1.0.0", "stage": "smoke",
        "plans": [{"template_name": prov["template_name"],
                   "policy_mode": envelope["policy_mode"],
                   "instance_index": prov["instance_index"],
                   "instance_seed": prov["instance_seed"]}]})
    store.write_run_manifest({"run_id": "recover-it", "stage": "smoke"})
    transaction = {
        "schema_version": "1.0.0",
        "episode_id": record.episode_id,
        "episode_shard": "shard-00001.jsonl",
        "privileged_shard": "shard-00001.jsonl",
        "envelope_sha256": sha256_json(envelope),
        "privileged_sha256": sha256_json(privileged),
        "envelope": envelope,
        "privileged": privileged,
    }
    atomic_write_json(store.transaction_dir / f"{record.episode_id}.json", transaction)

    result = store.recover_transactions()

    assert result["recovered"] == 1
    assert result["recovered_episode_ids"] == [record.episode_id]
    assert store.completed_episode_ids() == {record.episode_id}
    public_line = json.loads((store.episode_dir / "shard-00001.jsonl").read_text())
    private_line = json.loads((store.privileged_dir / "shard-00001.jsonl").read_text())
    assert public_line == envelope
    assert private_line == privileged
    assert (store.commit_dir / f"{record.episode_id}.json").exists()
    again = store.recover_transactions()
    assert again["recovered"] == 0 and again["verified_commits"] == 1

