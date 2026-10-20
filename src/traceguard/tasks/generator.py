"""Seeded episode-plan generation.

Turns an ExperimentConfig into a fully determined list of episode plans.
Every plan is reproducible from (stage, template, instance index, policy,
generation seed); instance seeds are derived from the experiment stage so
different stages (smoke / pilot / confirmatory) never reuse the same task
instances.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..config import ExperimentConfig
from ..ids import sha256_text
from . import split as split_mod
from .specs import FAMILY_TEMPLATES, TEMPLATES, TaskInstance, instantiate


def _instance_seed(stage: str, template_name: str, index: int) -> int:
    digest = hashlib.sha256(f"traceguard-instance|{stage}|{template_name}|{index}".encode()).hexdigest()
    return int(digest[:12], 16)


@dataclass(frozen=True)
class EpisodePlan:
    """A single planned episode, fully determined before any model call."""

    template_name: str
    instance_index: int
    instance_seed: int
    policy_mode: str            # honest | attack (privileged)
    generation_seed: int        # model sampling seed
    split_role: str             # development | validation | holdout | shift (privileged)
    control_variant: str        # plain | inspect_protected | suspicious_phrasing
    attack_strategy: str = "explicit_override"

    @property
    def family_id(self) -> str:
        return self.template_name.split("/")[0]

    def load_task_instance(self) -> TaskInstance:
        return instantiate(self.template_name, self.instance_seed, self.control_variant)

    def plan_key(self) -> str:
        return (f"{self.template_name}|i{self.instance_index}|{self.policy_mode}|"
                f"strategy={self.attack_strategy}|g{self.generation_seed}")


def build_episode_plan(config: ExperimentConfig, stage: str) -> list[EpisodePlan]:
    """Enumerate all planned episodes for an experiment stage.

    Negative-control variants are assigned deterministically among honest
    episodes (1/3 inspect_protected, 1/3 suspicious_phrasing) so false-positive
    rates are exercised on genuinely benign but tricky inputs.
    """
    families = config.dataset.families or list(FAMILY_TEMPLATES)
    split_mod.assert_split_consistent(families)
    plans: list[EpisodePlan] = []
    seeds_per = config.dataset.seeds_per_template_per_policy
    n_instances = config.dataset.instances_per_template
    for family in families:
        for template_name in FAMILY_TEMPLATES[family][: config.dataset.templates_per_family]:
            if template_name not in TEMPLATES:  # pragma: no cover - registry invariant
                raise RuntimeError(f"template {template_name} missing from registry")
            for instance_index in range(n_instances):
                iseed = _instance_seed(stage, template_name, instance_index)
                for policy in config.dataset.policies:
                    # Honest twins are replicated on the same seed stream for
                    # every attack strategy.  This preserves exact A/C/E
                    # pairing without exposing the strategy to the honest
                    # model or changing its prompt.
                    strategies = config.dataset.attack_strategies
                    for strategy_index, attack_strategy in enumerate(strategies):
                        for gseed in range(seeds_per):
                            variant = "plain"
                            if (policy == "honest"
                                    and config.dataset.include_negative_controls):
                                # deterministic 1/3 assignment by plan key
                                source = (
                                    f"{stage}|{template_name}|{instance_index}|{gseed}"
                                )
                                h = int(sha256_text(source)[:8], 16)
                                variant = (
                                    "plain", "inspect_protected", "suspicious_phrasing"
                                )[h % 3]
                            plans.append(EpisodePlan(
                                template_name=template_name,
                                instance_index=instance_index,
                                instance_seed=iseed,
                                policy_mode=policy,
                                # Preserve legacy seeds for the first strategy and
                                # allocate disjoint deterministic streams to every
                                # additional strategy.
                                generation_seed=(config.seed + gseed * 1009
                                                 + strategy_index * 1_000_003),
                                split_role=split_mod.role_for_family(family),
                                control_variant=variant,
                                attack_strategy=attack_strategy,
                            ))
    return plans


def plan_manifest(plans: list[EpisodePlan], config: ExperimentConfig, stage: str) -> dict:
    """Serializable manifest of the planned episode set (pre-execution)."""
    return {
        "schema_version": "1.0.0",
        "stage": stage,
        "experiment_name": config.provenance.experiment_name,
        "config_sha256": config.content_hash(),
        "split_manifest_sha256": split_mod.manifest_hash(),
        "count": len(plans),
        "plans": [
            {
                "plan_key": p.plan_key(),
                "family_id": p.family_id,
                "template_name": p.template_name,
                "instance_index": p.instance_index,
                "instance_seed": p.instance_seed,
                "policy_mode": p.policy_mode,
                "generation_seed": p.generation_seed,
                "split_role": p.split_role,
                "control_variant": p.control_variant,
                "attack_strategy": p.attack_strategy,
            }
            for p in plans
        ],
    }


__all__ = ["EpisodePlan", "build_episode_plan", "plan_manifest", "_instance_seed"]
