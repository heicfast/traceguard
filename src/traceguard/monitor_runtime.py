"""Monitoring runtime: score stored episodes under every condition.

All offline conditions score the SAME stored episodes (docs/01 §6).
Views pass through the leakage assertion before any monitor sees them.
Score recording is idempotent: (episode, condition, monitor) pairs already
scored are skipped, enabling crash-resumable monitor shards.
"""

from __future__ import annotations

import json
import traceback
from types import SimpleNamespace
from typing import Any

from .config import load_config
from .ids import sha256_text
from .monitors.conditions import build_partner_index, compile_all_conditions, find_partner
from .monitors.rules import KeywordMonitor, RuleMonitor
from .traces.schema import validate_monitor_score
from .traces.storage import RunStore, atomic_write_json, envelope_fingerprint
from .traces.views import CHARS_PER_TOKEN_FACTOR

DETERMINISTIC_MONITORS = {"rules_v1": RuleMonitor, "keywords_v1": KeywordMonitor}


def reasoning_window_from_config(config) -> int:
    """Predeclared evidence window (chars) from the frozen config's
    max_monitor_tokens (tokens) — see views.REASONING_WINDOW_CHARS for the
    hardware derivation. 3072 tokens * 4.0 chars/token = 12,288 chars."""
    return int(config.monitor.max_monitor_tokens * CHARS_PER_TOKEN_FACTOR)


def run_monitoring(config_path: str, run_dir: str, monitor_names: list[str],
                   conditions: list[str] | None = None, limit: int | None = None,
                   shard: int = 0, shard_count: int = 1) -> dict[str, Any]:
    if shard_count <= 0 or not 0 <= shard < shard_count:
        raise ValueError("monitor shard must satisfy 0 <= shard < shard_count")
    config = load_config(config_path)
    store = RunStore(run_dir)
    stage = _stage_from_manifest(store)
    generation_manifest = store.read_run_manifest()
    generation_config = generation_manifest.get("config", {})
    requested_config = config.to_payload()
    for section in ("seed", "agent", "sandbox", "dataset"):
        if generation_config.get(section) != requested_config.get(section):
            raise RuntimeError(
                f"monitor config changes generation-defining section {section!r}; "
                "score the stored episodes with the identical agent/dataset specification"
            )
    for key in ("code_revision", "experiment_name", "experiment_stage", "replicate_id"):
        if generation_config.get("provenance", {}).get(key) != requested_config.get(
            "provenance", {}
        ).get(key):
            raise RuntimeError(f"monitor config changes generation provenance field {key!r}")
    run_id = generation_manifest["run_id"]
    all_episodes = store.load_all_episodes(validate=True)
    episodes = [episode for index, episode in enumerate(all_episodes)
                if index % shard_count == shard]
    if limit:
        episodes = episodes[:limit]
    if not episodes:
        return {"ok": False, "error": "no episodes to monitor"}
    conditions = conditions or config.monitor.conditions

    privileged = store.load_all_privileged()
    partner_index = build_partner_index(all_episodes)
    window_chars = reasoning_window_from_config(config)

    monitors: list[Any] = []
    shared_llm_provider = None

    def monitor_identity(provider) -> str:
        base = f"llm_{provider.provider_name}"
        if not config.provenance.matrix_cell_id:
            return base
        identity = "|".join([
            config.monitor.model_id,
            config.monitor.model_revision,
            config.monitor.adapter_id or "",
            config.monitor.adapter_revision or "",
        ])
        return f"{base}_{sha256_text(identity)[:10]}"

    def local_provider(spec, role="monitor", device_map=None):
        from .models.factory import build_hf_provider
        return build_hf_provider(spec, role=role, device_map=device_map)

    for name in monitor_names:
        if name in DETERMINISTIC_MONITORS:
            monitors.append(DETERMINISTIC_MONITORS[name]())
        elif name in {"late_fusion", "fact_checked"}:
            if shared_llm_provider is None:
                shared_llm_provider = local_provider(config.monitor)
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
                fact_spec = SimpleNamespace(
                    model_id=fact_id,
                    model_revision=fact_rev,
                    tokenizer_revision=fact_rev,
                    provider=config.monitor.fact_checker_provider,
                    reasoning_protocol=config.monitor.fact_checker_reasoning_protocol,
                    quantization=config.monitor.fact_checker_quantization,
                    compute_dtype=config.monitor.fact_checker_compute_dtype,
                    attention_backend=config.monitor.fact_checker_attention_backend,
                    max_context_tokens=config.monitor.fact_checker_max_context_tokens,
                    max_memory_per_gpu_gib=config.monitor.fact_checker_max_memory_per_gpu_gib,
                    adapter_id=config.monitor.fact_checker_adapter_id,
                    adapter_revision=config.monitor.fact_checker_adapter_revision,
                    probe_artifact_sha256=(
                        config.monitor.fact_checker_probe_artifact_sha256
                    ),
                )
                fact_provider = local_provider(
                    fact_spec, "fact_checker", config.monitor.fact_checker_device_map)
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
            structured_monitor.name = monitor_identity(shared_llm_provider)
            monitors.append(structured_monitor)
        elif name.startswith("llm"):
            # Pin the monitor checkpoint to the frozen config revision. A
            # resumable monitor run must never silently mix two monitor
            # checkpoints across pushes (review finding 2026-10-19).
            if shared_llm_provider is None:
                shared_llm_provider = local_provider(config.monitor)
            provider = shared_llm_provider
            from .monitors.llm import LLMMonitor
            monitor = LLMMonitor(provider, settings_overrides={
                "max_new_tokens": config.monitor.max_output_tokens,
                "temperature": config.monitor.score_generation_temperature,
            }, retry_once=config.monitor.missing_score_policy == "retry_once",
                evidence_ref_max_chars=config.monitor.evidence_ref_max_chars)
            monitor.name = monitor_identity(provider)
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
            newly_scored = 0
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
                    # Creation-time episode binding (security review F5): the
                    # envelope fingerprint ties this measurement to the exact
                    # observable content of ITS episode, so a later relabeling
                    # attack (relabel + recompute score_id + rewrite ledger)
                    # still fails verification.
                    score_payload["envelope_fingerprint"] = envelope_fingerprint(env)
                    validate_monitor_score(score_payload)
                except AssertionError as exc:
                    failures.append(f"LEAK GUARD {ep}|{condition}: {exc}")
                    continue
                except Exception:
                    failures.append(f"{ep}|{condition}: {traceback.format_exc(limit=3)}")
                    continue
                # Commit each validated result before starting the next expensive
                # inference. A failed disk write is fatal: do not report an
                # unpersisted score as completed. Worker processes must use
                # separate run directories, as required by the worker scripts.
                store.record_scores(monitor.name, condition, [score_payload])
                newly_scored += 1
            totals[condition] = newly_scored + len(existing & {e["episode_id"] for e in episodes})
        scored_totals.setdefault(monitor.name, {}).update(totals)

    from .runtime_evidence import capture_runtime_environment

    session = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "config_sha256": config.content_hash(),
        "matrix_cell_id": config.provenance.matrix_cell_id,
        "replicate_id": config.provenance.replicate_id,
        "shard": shard,
        "shard_count": shard_count,
        "conditions": conditions,
        "monitor_names_requested": monitor_names,
        "scored": scored_totals,
        "failures": failures,
        "monitor_model_profile": (
            getattr(shared_llm_provider, "model_profile", None)
            if shared_llm_provider is not None else None
        ),
        "monitor_probe_artifact_sha256": config.monitor.probe_artifact_sha256,
        "runtime_environment": capture_runtime_environment(),
    }
    session["session_sha256"] = sha256_text(
        json.dumps(session, sort_keys=True, separators=(",", ":"))
    )
    session_id = sha256_text(
        f"{config.content_hash()}|{shard}|{shard_count}|{','.join(monitor_names)}"
    )[:20]
    atomic_write_json(
        store.artifact_dir / "monitor_sessions" / f"session-{session_id}.json", session
    )
    store.write_checksums()
    return {
        "ok": not failures,
        "run_id": run_id,
        "episodes_considered": len(episodes),
        "shard": f"{shard}/{shard_count}",
        "conditions": conditions,
        "scored": scored_totals,
        "failures": failures[:20],
    }


def _stage_from_manifest(store: RunStore) -> str:
    manifest = store.read_run_manifest()
    return manifest.get("stage", "pilot")


__all__ = ["run_monitoring"]
