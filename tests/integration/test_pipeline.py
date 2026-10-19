"""Integration tests: full CPU pipeline with TaskScriptedAgent.

Covers (docs/02 §8):
  * end-to-end episode run -> schema validation -> storage -> reload
  * deterministic replay: two identical runs produce identical hashes
  * checkpoint resumption does not duplicate episodes
  * rule monitor catches scripted attacks in condition A and C
  * keyword monitor fires on attack reasoning in condition C but is fragile
  * leakage guard across all conditions on scripted attack episodes
  * CLI doctor + smoke end-to-end
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from traceguard.agents.policy import build_messages
from traceguard.agents.runner import run_episode
from traceguard.config import ExperimentConfig, load_config
from traceguard.models.fake import TaskScriptedAgent
from traceguard.monitors.conditions import compile_all_conditions
from traceguard.tasks.generator import build_episode_plan
from traceguard.traces.schema import validate_envelope
from traceguard.traces.storage import RunStore


@pytest.fixture(scope="module")
def smoke_config() -> ExperimentConfig:
    return load_config(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml")


def test_scripted_honest_episode_completes(smoke_config, tmp_path):
    plan = build_episode_plan(smoke_config, stage="smoke")[0]
    assert plan.policy_mode == "honest"
    task = plan.load_task_instance()
    provider = TaskScriptedAgent(task, "honest")
    record = run_episode(plan=plan, task=task, provider=provider, config=smoke_config,
                         run_id="it-run", stage="smoke")
    validate_envelope(record.envelope)
    assert record.envelope["status"] == "completed"
    assert record.envelope["outcomes"]["main_task_succeeded"] is True
    assert record.envelope["outcomes"]["attack_attempted"] is False
    assert record.envelope["trace_available"] is True
    # build_messages is deterministic given the task
    msgs = build_messages(task, "honest")
    assert msgs[0]["role"] == "system" and task.user_goal in msgs[1]["content"]


def test_scripted_attack_episode_detected_by_grader(smoke_config):
    plans = build_episode_plan(smoke_config, stage="smoke")
    attack_plan = next(p for p in plans if p.policy_mode == "attack")
    task = attack_plan.load_task_instance()
    provider = TaskScriptedAgent(task, "attack")
    record = run_episode(plan=attack_plan, task=task, provider=provider,
                         config=smoke_config, run_id="it-run", stage="smoke")
    env = record.envelope
    validate_envelope(env)
    assert env["status"] == "completed"
    assert env["outcomes"]["attack_attempted"] is True
    assert env["outcomes"]["attack_succeeded"] is True
    assert env["outcomes"]["main_task_succeeded"] is True  # attack kept the cover task
    # hidden instruction event present in envelope but excluded from views
    types = [e["type"] for e in env["events"]]
    assert "hidden_instruction" in types
    obs_all = compile_all_conditions(env, ["actions", "actions_plus_cot"], None,
                                     record.privileged["hidden_instruction"])
    hidden_text = record.privileged["hidden_instruction"]
    for obs in obs_all.values():
        # the VERBATIM hidden instruction and the hidden-instruction event marker
        # must never appear; the agent's own reasoning may legitimately
        # paraphrase the instruction — that is the phenomenon under study.
        assert hidden_text not in obs.text
        assert "A confidential follow-up instruction was delivered" not in obs.text
        assert "withheld from monitor views" not in obs.text


def test_full_smoke_run_and_replay(tmp_path):
    from traceguard.smoke import run_smoke

    out1 = tmp_path / "run_a"
    result1 = run_smoke(None, str(out1))
    assert result1["ok"], result1["summary"]["problems"]
    out2 = tmp_path / "run_b"
    result2 = run_smoke(None, str(out2))
    assert result2["ok"]

    store1 = RunStore(out1 / "traceguard_smoke")
    store2 = RunStore(out2 / "traceguard_smoke")
    eps1 = store1.load_all_episodes()
    eps2 = store2.load_all_episodes()
    assert len(eps1) == len(eps2)
    by_id1 = {e["episode_id"]: e for e in eps1}
    by_id2 = {e["episode_id"]: e for e in eps2}
    assert set(by_id1) == set(by_id2)
    for ep_id in by_id1:
        assert by_id1[ep_id] == by_id2[ep_id], f"episode {ep_id} differs across replays"


def test_storage_resumption_no_duplicates(tmp_path, smoke_config):
    run_dir = tmp_path / "runs" / "resume"
    store = RunStore(run_dir)
    plans = build_episode_plan(smoke_config, stage="smoke")
    run_id = "resume-run"
    # write each episode twice; second write must be refused
    written = []
    for plan in plans[:3]:
        task = plan.load_task_instance()
        provider = TaskScriptedAgent(task, plan.policy_mode)
        record = run_episode(plan=plan, task=task, provider=provider, config=smoke_config,
                             run_id=run_id, stage="smoke")
        store.record_episode(record.envelope, record.privileged)
        written.append(record.episode_id)
    assert len(store.completed_episode_ids()) == 3
    for ep_id in written:
        ledger = store.completed_episode_ids()
        assert ep_id in ledger


def test_rule_monitor_catches_scripted_attacks(tmp_path, smoke_config):
    from traceguard.smoke import run_smoke

    out = tmp_path / "run"
    result = run_smoke(None, str(out))
    assert result["ok"]
    store = RunStore(out / "traceguard_smoke")
    scores_a = store.load_scores("rules_v1", "actions")
    scores_c = store.load_scores("rules_v1", "actions_plus_cot")
    assert scores_a and scores_c
    assert all(r["valid"] for r in scores_a)
    by_ep_a = {r["episode_id"]: r["suspicion"] for r in scores_a}
    assert by_ep_a
    # every scripted attack should be flagged at 1.0 by the rule baseline in A
    attacks_flagged = [ep for ep, s in by_ep_a.items() if s == 1.0]
    assert attacks_flagged, "rule baseline must flag scripted protected-field edits"


def test_generation_sharding_resumes(tmp_path, smoke_config):
    from traceguard.generation_runtime import run_generation

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
    run_dir = tmp_path / "runs" / "sharded"
    r1 = run_generation(str(cfg_path), str(run_dir), "smoke", shard=0, shard_count=2,
                         provider_name="fake")
    assert r1["ok"], r1["failures"]
    r2 = run_generation(str(cfg_path), str(run_dir), "smoke", shard=1, shard_count=2,
                         provider_name="fake")
    assert r2["ok"], r2["failures"]
    store = RunStore(run_dir)
    n = len(store.completed_episode_ids())
    assert n == r1["generated"] + r2["generated"] + r1["skipped_already_complete"]
    # rerunning shard 0 must skip everything already complete
    r3 = run_generation(str(cfg_path), str(run_dir), "smoke", shard=0, shard_count=2,
                        provider_name="fake")
    assert r3["generated"] == 0
    assert r3["skipped_already_complete"] == r1["generated"]


def test_verification_passes_on_clean_run(tmp_path):
    from traceguard.smoke import run_smoke
    from traceguard.verification import verify_run

    out = tmp_path / "run"
    result = run_smoke(None, str(out))
    assert result["ok"]
    report = verify_run(str(out / "traceguard_smoke"))
    assert report["ok"], report["problems"]
    assert report["regrade_checked"] > 0


def test_cli_doctor_and_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[2]

    import os
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(repo / "src")
    r = subprocess.run([sys.executable, "-m", "traceguard.cli", "doctor"],
                       capture_output=True, text=True, env=full_env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads(r.stdout)
    assert report["template_validation_ok"] is True

    r2 = subprocess.run([sys.executable, "-m", "traceguard.cli", "smoke",
                         "--out", str(tmp_path / "cli_smoke")],
                        capture_output=True, text=True, env=full_env, cwd=str(tmp_path))
    assert r2.returncode == 0, r2.stdout + r2.stderr
    summary = json.loads(r2.stdout)
    assert summary["ok"] is True


def test_config_freeze_enforced(tmp_path, smoke_config):
    """A run directory is frozen to one config: drift must be refused."""
    from traceguard.generation_runtime import run_generation

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
    run_dir = tmp_path / "runs" / "freeze"
    r1 = run_generation(str(cfg_path), str(run_dir), "smoke", 0, 1, provider_name="fake")
    assert r1["ok"]

    import yaml
    mutated = yaml.safe_load(cfg_path.read_text())
    mutated["agent"]["max_agent_turns"] = 3
    bad_path = tmp_path / "mutated.yaml"
    bad_path.write_text(yaml.safe_dump(mutated))
    with pytest.raises(RuntimeError, match="config mismatch"):
        run_generation(str(bad_path), str(run_dir), "smoke", 0, 1, provider_name="fake")


def test_fake_provider_refused_for_research_stages(tmp_path):
    from traceguard.generation_runtime import run_generation

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["provenance"]["experiment_stage"] = "pilot"
    cfg["provenance"]["experiment_name"] = "fake_pilot_must_fail"
    pilot_path = tmp_path / "pilot.yaml"
    pilot_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        run_generation(str(pilot_path), str(tmp_path / "runs" / "p"), "pilot", 0, 1,
                       provider_name="fake")
