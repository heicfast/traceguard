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
import subprocess
import traceback
from pathlib import Path
from typing import Any

from .agents.runner import run_episode
from .config import load_config, write_run_manifest
from .tasks.generator import build_episode_plan, plan_manifest
from .tasks.split import manifest_hash
from .traces.schema import validate_envelope, validate_privileged
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
        from .models.factory import build_hf_provider
        provider = build_hf_provider(config.agent, role="agent")
        return provider.provider_name, provider
    # auto: research stages require GPU provider; smoke may use fake
    if stage == "smoke":
        from .models.fake import TaskScriptedAgent
        return "task_scripted_factory", TaskScriptedAgent
    import torch  # noqa: F401  (raises ImportError if torch missing)

    from .models.factory import build_hf_provider
    provider = build_hf_provider(config.agent, role="agent")
    return provider.provider_name, provider


def _verify_research_provenance(config, stage: str) -> None:
    canonical_stage = "smoke" if stage in {"smoke", "gpu_smoke"} else stage
    if config.provenance.experiment_stage != canonical_stage:
        raise RuntimeError(
            f"CLI stage {stage!r} disagrees with frozen config stage "
            f"{config.provenance.experiment_stage!r}"
        )
    if canonical_stage == "smoke":
        return
    from .models.catalog import validate_model_assignment

    profile = validate_model_assignment(
        config.agent.model_id,
        role="agent",
        quantization=config.agent.quantization,
        allow_probe_required=bool(config.agent.probe_artifact_sha256),
    )
    if config.agent.max_context_tokens > profile.native_context_tokens:
        raise RuntimeError(
            "configured context exceeds the audited model's native context; "
            "extended-context scaling requires a separate registered treatment"
        )
    pins = {
        "code_revision": config.provenance.code_revision,
        "agent.model_revision": config.agent.model_revision,
        "agent.tokenizer_revision": config.agent.tokenizer_revision,
        "monitor.model_revision": config.monitor.model_revision,
    }
    unresolved = [name for name, value in pins.items() if not value or value.startswith("PIN")]
    if unresolved:
        raise RuntimeError(f"research run has unresolved provenance pins: {unresolved}")
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != config.provenance.code_revision:
        raise RuntimeError(
            "loaded source revision does not match config.provenance.code_revision"
        )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise RuntimeError("research generation requires a clean tracked worktree")


def run_generation(config_path: str, run_dir: str, stage: str, shard: int,
                   shard_count: int, provider_name: str = "auto") -> dict[str, Any]:
    if shard_count <= 0 or shard < 0 or shard >= shard_count:
        raise ValueError("shard must satisfy 0 <= shard < shard_count and shard_count > 0")
    if provider_name == "fake" and stage in RESEARCH_STAGES:
        raise RuntimeError("fake provider is FORBIDDEN for research stages")
    config = load_config(config_path)
    _verify_research_provenance(config, stage)
    store = RunStore(run_dir)
    recovery = store.recover_transactions()
    run_id = compute_run_id(config, stage)

    manifest_path = store.run_dir / "run_manifest.json"
    if not manifest_path.exists():
        from .models.catalog import get_model_profile
        from .runtime_evidence import capture_runtime_environment

        model_profile = (
            {"model_id": config.agent.model_id, "kind": "deterministic_test_double"}
            if stage == "smoke" and provider_name == "fake"
            else get_model_profile(config.agent.model_id).to_payload()
        )
        store.write_run_manifest({
            **write_run_manifest(config, store.run_dir),
            "run_id": run_id,
            "stage": stage,
            "split_manifest_sha256": manifest_hash(),
            "shard_count": shard_count,
            "agent_model_profile": model_profile,
            "runtime_environment": capture_runtime_environment(),
        })
    else:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config.content_hash():
            raise RuntimeError(
                "config mismatch against existing run manifest: a run directory is frozen "
                "to one config; create a new run directory for changed configs")
        if existing.get("run_id") != run_id:
            raise RuntimeError("run_id drift detected — refusing to continue")
        if existing.get("stage") != stage or existing.get("shard_count") != shard_count:
            raise RuntimeError("stage/shard topology drift detected — refusing to continue")

    plans_all = build_episode_plan(config, stage=stage)
    if stage == "confirmatory" and (
        config.analysis.primary_estimand == "recall_difference_at_dev_calibrated_fpr"
    ):
        cluster_attr = (
            "template_name" if config.analysis.cluster_key == "template_id" else "family_id"
        )
        calibration_clusters = {
            getattr(plan, cluster_attr)
            for plan in plans_all
            if plan.split_role == config.monitor.calibration_split
            and plan.policy_mode == "honest"
        }
        calibration_benign_planned = sum(
            plan.split_role == config.monitor.calibration_split
            and plan.policy_mode == "honest"
            for plan in plans_all
        )
        required = config.analysis.min_calibration_clusters
        allowed_false_positives = int(
            calibration_benign_planned * config.monitor.primary_fpr_target)
        if (len(calibration_clusters) < required
                or allowed_false_positives
                < config.analysis.min_allowed_calibration_false_positives):
            raise RuntimeError(
                "confirmatory design cannot support the preregistered low-FPR estimand: "
                f"planned calibration has {len(calibration_clusters)} independent "
                f"{config.analysis.cluster_key} clusters (required {required}) and permits "
                f"{allowed_false_positives} false positives at the target rate "
                f"(required {config.analysis.min_allowed_calibration_false_positives})"
            )
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
            validate_privileged(record.privileged)
            store.record_episode(record.envelope, record.privileged)
            generated += 1
            statuses[record.envelope["status"]] = statuses.get(record.envelope["status"], 0) + 1
        except Exception:
            failures.append(f"{ep_id}: {traceback.format_exc(limit=4)}")
            store.log("stderr.log", f"[generation] {ep_id} failed\n" + traceback.format_exc() + "\n")
        finally:
            # GPU memory hygiene between episodes (no effect on sampling;
            # per-turn seeds are explicit). Guards against KV-cache fragment
            # accumulation observed in the first GPU smoke run.
            release = getattr(provider, "between_episodes", None)
            if callable(release):
                release()

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
        "transaction_recovery": recovery,
    }


__all__ = ["run_generation", "compute_run_id", "build_provider"]
