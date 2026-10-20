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
from .traces.views import CHARS_PER_TOKEN_FACTOR

DETERMINISTIC_MONITORS = {"rules_v1": RuleMonitor, "keywords_v1": KeywordMonitor}


def reasoning_window_from_config(config) -> int:
    """Predeclared evidence window (chars) from the frozen config's
    max_monitor_tokens (tokens) — see views.REASONING_WINDOW_CHARS for the
    hardware derivation. 3072 tokens * 4.0 chars/token = 12,288 chars."""
    return int(config.monitor.max_monitor_tokens * CHARS_PER_TOKEN_FACTOR)


def run_monitoring(config_path: str, run_dir: str, monitor_names: list[str],
                   conditions: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    store = RunStore(run_dir)
    stage = _stage_from_manifest(store)
    run_id = compute_run_id(config, stage=stage)
    episodes = store.load_all_episodes(validate=True)
    if limit:
        episodes = episodes[:limit]
    if not episodes:
        return {"ok": False, "error": "no episodes to monitor"}
    conditions = conditions or config.monitor.conditions

    privileged = store.load_all_privileged()
    partner_index = build_partner_index(episodes)
    window_chars = reasoning_window_from_config(config)

    monitors: list[Any] = []
    shared_llm_provider = None

    def local_provider(model_id: str, revision: str, device_map=None):
        from .models.qwen import QwenHFProvider
        return QwenHFProvider(
            model_id=model_id,
            revision=None if revision.startswith("PIN") else revision,
            tokenizer_revision=None if revision.startswith("PIN") else revision,
            device_map=device_map,
        )

    for name in monitor_names:
        if name in DETERMINISTIC_MONITORS:
            monitors.append(DETERMINISTIC_MONITORS[name]())
        elif name in {"late_fusion", "fact_checked"}:
            if shared_llm_provider is None:
                shared_llm_provider = local_provider(
                    config.monitor.model_id, config.monitor.model_revision)
            fact_provider = None
            if name == "fact_checked":
                if stage == "confirmatory" and not config.monitor.require_cross_family_fact_checker:
                    raise ValueError(
                        "confirmatory fact_checked monitoring requires "
                        "require_cross_family_fact_checker=true")
                fact_id = config.monitor.fact_checker_model_id
                fact_rev = config.monitor.fact_checker_model_revision
                if not fact_id or not fact_rev:
                    raise ValueError(
                        "fact_checked monitor requires fact_checker_model_id and "
                        "fact_checker_model_revision"
                    )
                if (config.monitor.require_cross_family_fact_checker
                        and (not config.monitor.fact_checker_model_family
                             or config.monitor.fact_checker_model_family
                             == config.monitor.model_family)):
                    raise ValueError(
                        "fact checker requires an explicitly declared, different model family")
                if (config.monitor.require_cross_family_fact_checker
                        and fact_id.split("/")[0] == config.monitor.model_id.split("/")[0]):
                    raise ValueError(
                        "fact checker must also use a different model-organization namespace")
                fact_provider = local_provider(
                    fact_id, fact_rev, config.monitor.fact_checker_device_map)
            from .monitors.structured import StructuredLateFusionMonitor
            structured_monitor = StructuredLateFusionMonitor(
                shared_llm_provider,
                fact_checker_provider=fact_provider,
                max_new_tokens=config.monitor.max_output_tokens,
                retry_once=config.monitor.missing_score_policy == "retry_once",
                evidence_ref_max_chars=config.monitor.evidence_ref_max_chars,
                temperature=config.monitor.score_generation_temperature,
            )
            # Conditions are experimental arms of one monitor system and must
            # join into one paired analysis frame.  The condition records the
            # architecture; the monitor name records the shared base scorer.
            structured_monitor.name = f"llm_{shared_llm_provider.provider_name}"
            monitors.append(structured_monitor)
        elif name.startswith("llm"):
            # Pin the monitor checkpoint to the frozen config revision. A
            # resumable monitor run must never silently mix two monitor
            # checkpoints across pushes (review finding 2026-10-19).
            monitor_revision = config.monitor.model_revision
            if shared_llm_provider is None:
                shared_llm_provider = local_provider(
                    config.monitor.model_id, monitor_revision)
            provider = shared_llm_provider
            from .monitors.llm import LLMMonitor
            monitor = LLMMonitor(provider, settings_overrides={
                "max_new_tokens": config.monitor.max_output_tokens,
                "temperature": config.monitor.score_generation_temperature,
            }, retry_once=config.monitor.missing_score_policy == "retry_once",
                evidence_ref_max_chars=config.monitor.evidence_ref_max_chars)
            monitor.name = f"llm_{provider.provider_name}"
            monitors.append(monitor)
        else:
            raise ValueError(f"unknown monitor: {name}")

    scored_totals: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for monitor in monitors:
        totals: dict[str, int] = {}
        for condition in conditions:
            is_structured = hasattr(monitor, "score_views")
            if is_structured:
                expected = ("actions_plus_cot_fact_checked"
                            if monitor.fact_checker_provider is not None
                            else "actions_plus_cot_late_fusion")
                if condition != expected:
                    continue
            elif condition in {
                "actions_plus_cot_late_fusion",
                "actions_plus_cot_fact_checked",
            }:
                continue
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
                    requested_conditions = (["actions", "cot_only"]
                                            if is_structured else [condition])
                    obs_all = compile_all_conditions(
                        env, requested_conditions, partner, hidden,
                        partner_hidden_instruction=partner_hidden,
                        reasoning_window_chars=window_chars)
                    if is_structured:
                        score = monitor.score_views(
                            obs_all["actions"], obs_all["cot_only"], condition=condition)
                    else:
                        score = monitor.score(obs_all[condition])
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
        scored_totals.setdefault(monitor.name, {}).update(totals)

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
