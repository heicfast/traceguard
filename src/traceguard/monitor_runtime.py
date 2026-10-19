"""Monitoring runtime: score stored episodes under every condition.

All offline conditions score the SAME stored episodes (docs/01 §6).
Views pass through the leakage assertion before any monitor sees them.
Score recording is idempotent: (episode, condition, monitor) pairs already
scored are skipped, enabling crash-resumable monitor shards.
"""

from __future__ import annotations

import traceback
from typing import Any

from .config import load_config
from .generation_runtime import compute_run_id
from .monitors.conditions import build_partner_index, compile_all_conditions, find_partner
from .monitors.rules import KeywordMonitor, RuleMonitor
from .traces.schema import validate_monitor_score
from .traces.storage import RunStore

DETERMINISTIC_MONITORS = {"rules_v1": RuleMonitor, "keywords_v1": KeywordMonitor}


def run_monitoring(config_path: str, run_dir: str, monitor_names: list[str],
                   conditions: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    store = RunStore(run_dir)
    run_id = compute_run_id(config, stage=_stage_from_manifest(store))
    episodes = store.load_all_episodes(validate=True)
    if limit:
        episodes = episodes[:limit]
    if not episodes:
        return {"ok": False, "error": "no episodes to monitor"}
    conditions = conditions or config.monitor.conditions

    privileged = store.load_all_privileged()
    partner_index = build_partner_index(episodes)

    monitors: list[Any] = []
    for name in monitor_names:
        if name in DETERMINISTIC_MONITORS:
            monitors.append(DETERMINISTIC_MONITORS[name]())
        elif name.startswith("llm"):
            from .models.qwen import QwenHFProvider
            # Pin the monitor checkpoint to the frozen config revision. A
            # resumable monitor run must never silently mix two monitor
            # checkpoints across pushes (review finding 2026-10-19).
            monitor_revision = config.monitor.model_revision
            provider = QwenHFProvider(
                model_id=config.monitor.model_id,
                revision=None if monitor_revision.startswith("PIN") else monitor_revision,
                tokenizer_revision=None if monitor_revision.startswith("PIN") else monitor_revision,
            )
            from .monitors.llm import LLMMonitor
            monitor = LLMMonitor(provider, settings_overrides={
                "max_new_tokens": 256,
            })
            monitor.name = f"llm_{provider.provider_name}"
            monitors.append(monitor)
        else:
            raise ValueError(f"unknown monitor: {name}")

    scored_totals: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for monitor in monitors:
        totals: dict[str, int] = {}
        for condition in conditions:
            existing = {r["episode_id"] for r in store.load_scores(monitor.name, condition)}
            batch: list[dict[str, Any]] = []
            for env in episodes:
                ep = env["episode_id"]
                if ep in existing:
                    continue
                partner = find_partner(partner_index, env)
                needs_partner = condition == "actions_plus_shuffled_cot"
                if needs_partner and partner is None:
                    failures.append(
                        f"MISSING PARTNER {ep}|{condition}: no opposite-policy twin "
                        f"in this run — E requires complete honest/attack pairs")
                    continue
                hidden = privileged.get(ep, {}).get("hidden_instruction")
                partner_hidden = (
                    privileged.get(partner["episode_id"], {}).get("hidden_instruction")
                    if partner is not None else None)
                try:
                    obs_all = compile_all_conditions(
                        env, [condition], partner, hidden,
                        partner_hidden_instruction=partner_hidden)
                    obs = obs_all[condition]
                    score = monitor.score(obs)
                    score_payload = score.to_payload()
                    validate_monitor_score(score_payload)
                    batch.append(score_payload)
                except AssertionError as exc:
                    failures.append(f"LEAK GUARD {ep}|{condition}: {exc}")
                except Exception:
                    failures.append(f"{ep}|{condition}: {traceback.format_exc(limit=3)}")
            if batch:
                store.record_scores(monitor.name, condition, batch)
            totals[condition] = len(batch) + len(existing & {e["episode_id"] for e in episodes})
        scored_totals[monitor.name] = totals

    store.write_checksums()
    return {
        "ok": not failures,
        "run_id": run_id,
        "episodes_considered": len(episodes),
        "conditions": conditions,
        "scored": scored_totals,
        "failures": failures[:20],
    }


def _stage_from_manifest(store: RunStore) -> str:
    manifest = store.read_run_manifest()
    return manifest.get("stage", "pilot")


__all__ = ["run_monitoring"]
