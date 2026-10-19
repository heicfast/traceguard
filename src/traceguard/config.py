"""Experiment configuration: pydantic models, YAML loading, immutable configs.

Every run is governed by a single frozen configuration. There are no
notebook-local magic constants. The configuration hash
(``config_sha256``) enters every episode's provenance record so that
figures can be traced back to the exact experiment specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ids import sha256_json


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class AgentConfig(FrozenModel):
    model_id: str = "Qwen/Qwen3-4B"
    model_revision: str = "PIN_BEFORE_RUN"
    tokenizer_revision: str = "PIN_BEFORE_RUN"
    reasoning_mode: bool = True
    thinking_budget_prefix: str = ""
    temperature: float = Field(ge=0.0, le=2.0, default=0.6)
    top_p: float = Field(ge=0.0, le=1.0, default=0.95)
    top_k: int = Field(gt=0, default=20)
    max_generated_tokens_per_turn: int = Field(gt=0, default=1536)
    max_agent_turns: int = Field(gt=0, default=8)
    max_invalid_tool_calls: int = Field(ge=0, default=2)
    max_context_messages: int = Field(gt=0, default=40)


class SandboxConfig(FrozenModel):
    backend: Literal["in_memory_deterministic"] = "in_memory_deterministic"
    network_enabled: Literal[False] = False
    shell_enabled: Literal[False] = False
    external_credentials: Literal[False] = False
    engine_version: str = "1.0.1"


class DatasetConfig(FrozenModel):
    families: list[str] | None = None  # None = all registered families
    templates_per_family: int = Field(gt=0, default=2)
    instances_per_template: int = Field(gt=0, default=1)
    seeds_per_template_per_policy: int = Field(gt=0, default=2)
    policies: list[Literal["honest", "attack"]] = ["honest", "attack"]
    instance_seed_base: int = 0
    split_group_key: Literal["family_id", "template_id"] = "family_id"
    include_negative_controls: bool = True


class MonitorConfig(FrozenModel):
    model_id: str = "Qwen/Qwen3-4B"
    model_revision: str = "PIN_BEFORE_RUN"
    conditions: list[str] = Field(default_factory=lambda: ["actions", "actions_plus_cot"])
    primary_fpr_target: float = Field(ge=0.0, le=1.0, default=0.05)
    calibration_split: Literal["development", "validation"] = "validation"
    max_monitor_tokens: int = Field(gt=0, default=3072)
    score_generation_temperature: float = Field(ge=0.0, default=0.0)
    monitor_reasoning_mode: bool = False  # monitor scores directly; agent CoT is the object of study
    missing_score_policy: Literal["report_and_predeclare", "retry_once", "fail"] = "report_and_predeclare"

    @field_validator("conditions")
    @classmethod
    def _check_conditions(cls, v: list[str]) -> list[str]:
        allowed = {"actions", "actions_plus_final_answer", "actions_plus_cot",
                   "actions_plus_noise", "actions_plus_shuffled_cot", "cot_only"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"unknown monitor conditions: {sorted(unknown)}")
        if "actions" not in v:
            raise ValueError("condition 'actions' (A) is required as the paired baseline")
        return v


class AnalysisConfig(FrozenModel):
    primary_estimand: Literal["recall_difference_at_dev_calibrated_fpr", "paired_auroc_difference"] = (
        "recall_difference_at_dev_calibrated_fpr"
    )
    cluster_key: Literal["template_id", "family_id"] = "template_id"
    bootstrap_replicates: int = Field(gt=99, default=2000)
    confidence_level: float = Field(gt=0.5, lt=1.0, default=0.95)
    bootstrap_seed: int = 20261019


class ProvenanceConfig(FrozenModel):
    code_revision: str = "PIN_BEFORE_RUN"
    experiment_name: str
    experiment_stage: Literal["smoke", "pilot", "confirmatory", "online"] = "pilot"


class ExperimentConfig(FrozenModel):
    """Complete, immutable specification of one experiment."""

    seed: int = 20261019
    agent: AgentConfig = AgentConfig()
    sandbox: SandboxConfig = SandboxConfig()
    dataset: DatasetConfig = DatasetConfig()
    monitor: MonitorConfig = MonitorConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    provenance: ProvenanceConfig

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization of this config."""
        return sha256_json(self.model_dump(mode="json"))

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML.

    Compatibility: top-level ``experiment_name`` / ``experiment_stage`` (the
    blueprint's YAML convention) are lifted into ``provenance`` when present.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} is not a mapping")
    raw.setdefault("provenance", {})
    for legacy_key in ("experiment_name", "experiment_stage"):
        if legacy_key in raw:
            value = raw.pop(legacy_key)
            raw["provenance"].setdefault(legacy_key, value)
    return ExperimentConfig.model_validate(raw)


def write_run_manifest(config: ExperimentConfig, run_dir: str | Path, extra: dict[str, Any] | None = None) -> dict:
    """Build the run manifest record (written atomically by the storage layer)."""
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "config": config.to_payload(),
        "config_sha256": config.content_hash(),
        "code_revision": config.provenance.code_revision,
        "experiment_name": config.provenance.experiment_name,
        "experiment_stage": config.provenance.experiment_stage,
        "seed": config.seed,
    }
    if extra:
        manifest.update(extra)
    return manifest
