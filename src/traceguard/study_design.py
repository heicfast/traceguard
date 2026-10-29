"""Transparent study sizing and compute-budget planning.

The planner deliberately exposes assumptions instead of emitting a magical
episode count.  It combines a paired-binary power approximation, cluster
design effect, attack-attempt yield, monitor matrix size, and measured or
declared throughput.  It is a planning tool; the frozen preregistration owns
the final design.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import yaml

from .config import ExperimentConfig
from .model_matrix import build_schedule, load_model_matrix
from .tasks.generator import build_episode_plan


@dataclass(frozen=True)
class PlanningAssumptions:
    alpha: float = 0.05
    power: float = 0.80
    minimum_detectable_recall_difference: float = 0.10
    discordant_pair_rate: float = 0.30
    intracluster_correlation: float = 0.10
    mean_episodes_per_cluster: float = 8.0
    attack_policy_fraction: float = 0.50
    attack_attempt_rate: float = 0.60
    valid_paired_score_rate: float = 0.95
    generated_tokens_per_episode: int = 4_000
    monitor_tokens_per_episode_condition: int = 2_500
    generation_tokens_per_second: float = 12.0
    monitoring_tokens_per_second: float = 18.0
    gpu_count_per_job: int = 1
    operational_reserve_fraction: float = 0.20
    bytes_per_episode_raw: int = 500_000
    bytes_per_monitor_score: int = 80_000

    def validate(self) -> None:
        probabilities = {
            "alpha": self.alpha,
            "power": self.power,
            "minimum_detectable_recall_difference": (self.minimum_detectable_recall_difference),
            "discordant_pair_rate": self.discordant_pair_rate,
            "intracluster_correlation": self.intracluster_correlation,
            "attack_policy_fraction": self.attack_policy_fraction,
            "attack_attempt_rate": self.attack_attempt_rate,
            "valid_paired_score_rate": self.valid_paired_score_rate,
            "operational_reserve_fraction": self.operational_reserve_fraction,
        }
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if self.minimum_detectable_recall_difference >= self.discordant_pair_rate:
            raise ValueError("effect size must be smaller than discordant-pair rate")
        for name in (
            "mean_episodes_per_cluster",
            "generation_tokens_per_second",
            "monitoring_tokens_per_second",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mean_episodes_per_cluster < 1:
            raise ValueError("mean_episodes_per_cluster must be at least 1")
        for name in ("generated_tokens_per_episode", "monitor_tokens_per_episode_condition",
                     "gpu_count_per_job", "bytes_per_episode_raw", "bytes_per_monitor_score"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def paired_binary_sample_size(
    assumptions: PlanningAssumptions,
) -> dict[str, float | int | str]:
    """Approximate pairs required for a two-sided paired binary contrast.

    For discordant probabilities p10 and p01, Var(p10-p01) is approximately
    ``(q - delta^2) / n`` where q=p10+p01.  We inflate the independent-pair
    count by the standard exchangeable-cluster design effect.
    """
    assumptions.validate()
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - assumptions.alpha / 2.0)
    z_power = normal.inv_cdf(assumptions.power)
    delta = assumptions.minimum_detectable_recall_difference
    q = assumptions.discordant_pair_rate
    independent = math.ceil(((z_alpha + z_power) ** 2) * (q - delta * delta) / (delta * delta))
    design_effect = (
        1.0 + (assumptions.mean_episodes_per_cluster - 1.0) * assumptions.intracluster_correlation
    )
    clustered = math.ceil(independent * design_effect)
    return {
        "independent_attempted_positive_pairs": independent,
        "cluster_design_effect": design_effect,
        "cluster_adjusted_attempted_positive_pairs": clustered,
        "method": "normal_approximation_paired_binary_with_cluster_design_effect",
    }


def build_study_plan(
    config: ExperimentConfig,
    assumptions: PlanningAssumptions,
    *,
    matrix_path: str | None = None,
    stage: str = "pilot",
) -> dict[str, Any]:
    """Build a machine-readable workload, budget, and sizing plan."""
    sizing = paired_binary_sample_size(assumptions)
    yield_rate = (
        assumptions.attack_policy_fraction
        * assumptions.attack_attempt_rate
        * assumptions.valid_paired_score_rate
    )
    adjusted_pairs = sizing["cluster_adjusted_attempted_positive_pairs"]
    if not isinstance(adjusted_pairs, int):
        raise TypeError("cluster-adjusted pair estimate must be an integer")
    required_episodes = math.ceil(adjusted_pairs / yield_rate)
    configured_episodes = len(build_episode_plan(config, stage))
    conditions = len(config.monitor.conditions)

    matrix_validation_error: str | None = None
    if matrix_path:
        try:
            schedule = build_schedule(load_model_matrix(matrix_path))
            monitoring_jobs = schedule.get("monitoring_jobs", [])
            scheduled_generation_jobs = schedule.get("generation_jobs", [])
            matrix_cells = len(monitoring_jobs)
            generation_jobs = max(1, len(scheduled_generation_jobs))
            matrix_status = "validated_frozen_matrix"
        except ValueError as exc:
            # Operational matrix ledgers may contain status/output annotations
            # and pending revision placeholders. They are not executable, but
            # their declared shape is still useful for a clearly labeled draft
            # budget. Never silently treat this fallback as an admitted matrix.
            raw = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
            cells = raw.get("cells", []) if isinstance(raw, dict) else []
            if not isinstance(cells, list) or not cells:
                raise ValueError("matrix has no declared cells to estimate") from exc
            monitor_keys: set[tuple[str, int]] = set()
            generation_keys: set[tuple[str, int]] = set()
            for cell in cells:
                if not isinstance(cell, dict):
                    raise ValueError("matrix cell is not a mapping") from exc
                cell_id = str(cell.get("cell_id", "missing-cell-id"))
                cohort = str(cell.get("generation_cohort_id", cell_id))
                replicate_ids = cell.get("replicate_ids", [0])
                if not isinstance(replicate_ids, list):
                    raise ValueError(f"matrix cell {cell_id} replicate_ids is not a list") from exc
                for replicate_id in replicate_ids:
                    monitor_keys.add((cell_id, int(replicate_id)))
                    generation_keys.add((cohort, int(replicate_id)))
            matrix_cells = len(monitor_keys)
            generation_jobs = max(1, len(generation_keys))
            matrix_status = "draft_shape_only_not_executable"
            matrix_validation_error = str(exc)
    else:
        matrix_cells = generation_jobs = 1
        matrix_status = "single_config_no_matrix"

    planned_episodes = max(required_episodes, configured_episodes)
    generation_tokens = (
        planned_episodes * assumptions.generated_tokens_per_episode * generation_jobs
    )
    monitor_scores = planned_episodes * conditions * matrix_cells
    monitoring_tokens = monitor_scores * assumptions.monitor_tokens_per_episode_condition
    reserve = 1.0 + assumptions.operational_reserve_fraction
    # Throughput describes one complete job, not each GPU in that job.
    # GPU-hours are wall time multiplied by the number of GPUs assigned.
    generation_gpu_hours = (generation_tokens / assumptions.generation_tokens_per_second / 3600
                            * assumptions.gpu_count_per_job)
    monitoring_gpu_hours = (monitoring_tokens / assumptions.monitoring_tokens_per_second / 3600
                            * assumptions.gpu_count_per_job)
    total_gpu_hours = (generation_gpu_hours + monitoring_gpu_hours) * reserve
    storage_bytes = (
        planned_episodes * assumptions.bytes_per_episode_raw * generation_jobs
        + monitor_scores * assumptions.bytes_per_monitor_score
    ) * reserve
    return {
        "schema_version": "1.0.0",
        "status": "planning_estimate_not_preregistration",
        "assumptions": asdict(assumptions),
        "sizing": {
            **sizing,
            "effective_positive_yield_per_episode": yield_rate,
            "required_total_episodes": required_episodes,
            "configured_episode_count": configured_episodes,
            "planned_episode_count": planned_episodes,
        },
        "matrix": {
            "status": matrix_status,
            "validation_error": matrix_validation_error,
            "cells": matrix_cells,
            "deduplicated_generation_cohorts": generation_jobs,
            "monitor_conditions_per_cell": conditions,
        },
        "workload": {
            "generation_tokens_upper_estimate": generation_tokens,
            "monitoring_tokens_upper_estimate": monitoring_tokens,
            "monitor_score_records": monitor_scores,
            "generation_gpu_hours_before_reserve": generation_gpu_hours,
            "monitoring_gpu_hours_before_reserve": monitoring_gpu_hours,
            "total_gpu_hours_with_reserve": total_gpu_hours,
            "storage_gib_with_reserve": storage_bytes / (1024**3),
            "gpu_count_per_job": assumptions.gpu_count_per_job,
        },
        "interpretation": [
            "Replace throughput and yield assumptions with pilot measurements before freeze.",
            "Power targets attempted positive pairs; low attack-attempt yield increases total episodes.",
            "The approximation must be validated by simulation for the frozen cluster distribution.",
            "GPU-hour totals are workload estimates, not wall-clock promises.",
            "Throughput assumptions are per job; GPU-hours include all GPUs assigned to the job.",
        ],
    }


__all__ = ["PlanningAssumptions", "paired_binary_sample_size", "build_study_plan"]
