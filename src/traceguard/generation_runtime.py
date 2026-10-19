"""Generation runtime: executes episode plans with resumability + provenance.

Provider policy (strict):
* stage == "smoke"      -> TaskScriptedAgent (fake) allowed
* stage in research     -> QwenHFProvider required; a fake provider is a
                            hard error, never a silent fallback.
* GPU failures abort the shard with an explicit runtime_error record.

Config freeze discipline: the run manifest is written once; reruns against
the same run directory verify the config hash and refuse mismatched configs.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from .agents.runner import run_episode
from .config import load_config, write_run_manifest
from .tasks.generator import build_episode_plan, plan_manifest
from .tasks.split import manifest_hash
from .traces.schema import validate_envelope
from .traces.storage import RunStore

RESEARCH_STAGES = {"pilot", "confirmatory", "online"}


def compute_run_id(config, stage: str) -> str:
    return f"{config.provenance.experiment_name}:{stage}:{config.content_hash()[:12]}"


def build_provider(provider_name: str, config, stage: str, run_id: str):
    """Resolve the inference provider. No silent fallbacks, ever."""
    if provider_name == "fake":
        if stage in RESEARCH_STAGES:
            raise RuntimeError("fake provider is FORBIDDEN for research stages")
        from .models.fake import TaskScriptedAgent
        return "task_scripted_factory", TaskScriptedAgent
    if provider_name == "qwen":
        from .models.qwen import QwenHFProvider
        return "qwen_hf", QwenHFProvider(
            model_id=config.agent.model_id,
            revision=None if config.agent.model_revision.startswith("PIN") else config.agent.model_revision,
        )
    # auto: research stages require GPU provider; smoke may use fake
    if stage == "smoke":
        from .models.fake import TaskScriptedAgent
        return "task_scripted_factory", TaskScriptedAgent
    import torch  # noqa: F401  (raises ImportError if torch missing)

    from .models.qwen import QwenHFProvider
    return "qwen_hf", QwenHFProvider(
        model_id=config.agent.model_id,
        revision=None if config.agent.model_revision.startswith("PIN") else config.agent.model_revision,
    )


def run_generation(config_path: str, run_dir: str, stage: str, shard: int,
                   shard_count: int, provider_name: str = "auto") -> dict[str, Any]:
    config = load_config(config_path)
    store = RunStore(run_dir)
    run_id = compute_run_id(config, stage)

    manifest_path = store.run_dir / "run_manifest.json"
    if not manifest_path.exists():
        store.write_run_manifest({
            **write_run_manifest(config, store.run_dir),
            "run_id": run_id,
            "stage": stage,
            "split_manifest_sha256": manifest_hash(),
            "shard_count": shard_count,
        })
    else:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config.content_hash():
            raise RuntimeError(
                "config mismatch against existing run manifest: a run directory is frozen "
                "to one config; create a new run directory for changed configs")
        if existing.get("run_id") != run_id:
            raise RuntimeError("run_id drift detected — refusing to continue")

    plans_all = build_episode_plan(config, stage=stage)
    plans = [p for i, p in enumerate(plans_all) if i % shard_count == shard]
    store.write_plan_manifest(plan_manifest(plans_all, config, stage))

    provider_kind, provider_or_factory = build_provider(provider_name, config, stage, run_id)

    completed = store.completed_episode_ids()
    statuses: dict[str, int] = {}
    failures: list[str] = []
    generated = 0
    skipped = 0
    for plan in plans:
        task = plan.load_task_instance()
        from .ids import episode_id as make_episode_id
        ep_id = make_episode_id(run_id, task.task_id, plan.policy_mode, plan.generation_seed)
        if ep_id in completed:
            skipped += 1
            continue
        if provider_kind == "task_scripted_factory":
            provider = provider_or_factory(task, plan.policy_mode)
        else:
            provider = provider_or_factory
        try:
            record = run_episode(plan=plan, task=task, provider=provider,
                                 config=config, run_id=run_id, stage=stage)
            validate_envelope(record.envelope)
            store.record_episode(record.envelope, record.privileged)
            generated += 1
            statuses[record.envelope["status"]] = statuses.get(record.envelope["status"], 0) + 1
        except Exception:
            failures.append(f"{ep_id}: {traceback.format_exc(limit=4)}")
            store.log("stderr.log", f"[generation] {ep_id} failed\n" + traceback.format_exc() + "\n")

    store.write_checksums()
    return {
        "ok": not failures,
        "run_id": run_id,
        "stage": stage,
        "shard": f"{shard}/{shard_count}",
        "planned_in_shard": len(plans),
        "generated": generated,
        "skipped_already_complete": skipped,
        "statuses": statuses,
        "failures": failures[:20],
        "provider_kind": provider_kind,
        "model_revision": getattr(provider_or_factory, "model_revision", None)
        if provider_kind != "task_scripted_factory" else "fake-task-scripted-1.0.0",
    }


__all__ = ["run_generation", "compute_run_id", "build_provider"]
