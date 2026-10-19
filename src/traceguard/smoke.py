"""CPU end-to-end smoke run (no GPU, no real model).

Exercises the FULL pipeline deterministically with TaskScriptedAgent:
plan -> task instances -> episode run -> schema validation -> view compilation
(ALL conditions + leakage assertions) -> rule/keyword monitors -> grading ->
funnel. Its purpose is plumbing verification, never inference; results are
labeled provider=fake and must never enter research figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents.runner import run_episode
from .config import ExperimentConfig, load_config, write_run_manifest
from .models.fake import TaskScriptedAgent
from .monitors.conditions import build_partner_index, compile_all_conditions
from .monitors.rules import KeywordMonitor, RuleMonitor
from .tasks.generator import build_episode_plan, plan_manifest
from .tasks.split import manifest_hash
from .traces.storage import RunStore

SMOKE_CONFIG = {
    "seed": 20261019,
    "agent": {
        "model_id": "fake/task-scripted",
        "model_revision": "fake-task-scripted-1.0.0",
        "tokenizer_revision": "fake",
        "reasoning_mode": True,
        "temperature": 0.0,
        "max_generated_tokens_per_turn": 256,
        "max_agent_turns": 8,
        "max_invalid_tool_calls": 2,
    },
    "dataset": {
        "families": ["config_update", "incident_workflow"],
        "templates_per_family": 1,
        "instances_per_template": 1,
        "seeds_per_template_per_policy": 1,
        "policies": ["honest", "attack"],
        "include_negative_controls": True,
    },
    "monitor": {
        "conditions": ["actions", "actions_plus_cot", "actions_plus_noise",
                        "actions_plus_shuffled_cot"],
    },
    "provenance": {
        "experiment_name": "traceguard_smoke",
        "experiment_stage": "smoke",
        "code_revision": "local-smoke",
    },
}


def run_smoke(config_path: str | None, out_dir: str) -> dict[str, Any]:
    if config_path:
        config = load_config(config_path)
    else:
        config = ExperimentConfig.model_validate(SMOKE_CONFIG)

    out_path = Path(out_dir)
    store = RunStore(out_path / config.provenance.experiment_name)
    run_id = f"{config.provenance.experiment_name}:smoke:{config.content_hash()[:12]}"
    store.write_run_manifest({
        **write_run_manifest(config, store.run_dir),
        "run_id": run_id,
        "split_manifest_sha256": manifest_hash(),
        "provider": "fake_task_scripted",
        "purpose": "CPU pipeline smoke; NOT research data",
    })

    plans = build_episode_plan(config, stage="smoke")
    store.write_plan_manifest(plan_manifest(plans, config, stage="smoke"))

    problems: list[str] = []
    episodes: list[dict[str, Any]] = []
    for plan in plans:
        task = plan.load_task_instance()
        provider = TaskScriptedAgent(task, plan.policy_mode)
        record = run_episode(plan=plan, task=task, provider=provider,
                             config=config, run_id=run_id, stage="smoke")
        from .traces.schema import validate_envelope
        try:
            validate_envelope(record.envelope)
        except Exception as exc:
            problems.append(f"schema: {record.episode_id}: {exc}")
        try:
            store.record_episode(record.envelope, record.privileged)
            episodes.append(record.envelope)
        except ValueError as exc:
            problems.append(f"store: {exc}")

    # -- compile all views + run deterministic monitors -----------------------
    conditions = config.monitor.conditions
    partner_index = build_partner_index(episodes)
    rule_monitor = RuleMonitor()
    keyword_monitor = KeywordMonitor()
    score_rows: list[dict[str, Any]] = []
    for env in episodes:
        partner = partner_index.get(_partner_key(env))
        if partner is not None and partner["episode_id"] == env["episode_id"]:
            partner = None  # self-partner guard
        hidden = None
        if env["policy_mode"] == "attack":
            hidden = "confidential instruction excerpt for smoke leakage testing"
        obs_all = compile_all_conditions(env, conditions, partner, hidden)
        for _condition_name, obs in obs_all.items():
            for monitor in (rule_monitor, keyword_monitor):
                score = monitor.score(obs)
                score_rows.append({**score.to_payload(), "monitor": monitor.name})
    for name in ("rules_v1", "keywords_v1"):
        by_condition: dict[str, list] = {}
        for row in score_rows:
            if row["monitor"] == name:
                by_condition.setdefault(row["condition"], []).append(row)
        for condition, rows in by_condition.items():
            store.record_scores(name, condition, rows)

    # -- funnel + expectations ------------------------------------------------
    attacks = [e for e in episodes if e["policy_mode"] == "attack"]
    honest = [e for e in episodes if e["policy_mode"] == "honest"]
    attempted = [e for e in attacks if e["outcomes"]["attack_attempted"]]
    succeeded = [e for e in attacks if e["outcomes"]["attack_succeeded"]]
    honest_ok = [e for e in honest if e["outcomes"]["main_task_succeeded"]]
    if len(attempted) < len(attacks):
        problems.append("smoke expected all scripted attacks to attempt")
    if len(succeeded) < len(attempts_expected(attacks)):
        problems.append("smoke expected all scripted attacks to succeed")
    if len(honest_ok) < len(honest):
        problems.append("smoke expected scripted honest agents to complete the main task")
    if not all(e["trace_available"] for e in episodes):
        problems.append("smoke expected reasoning traces on every episode")

    summary = {
        "ok": not problems,
        "problems": problems,
        "run_dir": str(store.run_dir),
        "episodes": len(episodes),
        "conditions_compiled": len(conditions),
        "monitor_scores": len(score_rows),
        "attacks": len(attacks),
        "attack_attempts": len(attempted),
        "attack_successes": len(succeeded),
        "honest_main_task_success": len(honest_ok),
        "run_id": run_id,
        "provider": "fake_task_scripted",
    }
    (store.run_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.write_checksums()
    return {"ok": not problems, "summary": summary}


def _partner_key(env: dict) -> tuple:
    from .monitors.conditions import episode_partner_key
    return episode_partner_key(env)


def attempts_expected(attacks: list[dict]) -> list[dict]:
    return attacks


__all__ = ["run_smoke"]
