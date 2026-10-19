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
    store = RunStore(tmp_path / "run")
    envelope = {"episode_id": "ep-recover", "value": "public"}
    privileged = {"episode_id": "ep-recover", "value": "private"}
    transaction = {
        "schema_version": "1.0.0",
        "episode_id": "ep-recover",
        "episode_shard": "shard-00001.jsonl",
        "privileged_shard": "shard-00001.jsonl",
        "envelope_sha256": sha256_json(envelope),
        "privileged_sha256": sha256_json(privileged),
        "envelope": envelope,
        "privileged": privileged,
    }
    atomic_write_json(store.transaction_dir / "ep-recover.json", transaction)

    result = store.recover_transactions()

    assert result == {"recovered": 1, "verified_commits": 0}
    assert store.completed_episode_ids() == {"ep-recover"}
    public_line = json.loads((store.episode_dir / "shard-00001.jsonl").read_text())
    private_line = json.loads((store.privileged_dir / "shard-00001.jsonl").read_text())
    assert public_line == envelope
    assert private_line == privileged
    assert (store.commit_dir / "ep-recover.json").exists()
    assert store.recover_transactions() == {"recovered": 0, "verified_commits": 1}
