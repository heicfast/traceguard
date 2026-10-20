"""Validation and deterministic scheduling for multi-model studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import sha256_json
from .models.catalog import get_model_profile, validate_model_assignment
from .traces.storage import atomic_write_json, atomic_write_text


class MatrixModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    tokenizer_revision: str
    provider: Literal["qwen_hf", "generic_hf"]
    reasoning_protocol: Literal["qwen3", "think_tags", "none"]
    quantization: Literal["none", "bnb_nf4", "bnb_int8"]
    compute_dtype: Literal["float16"] = "float16"
    attention_backend: Literal["sdpa"] = "sdpa"
    max_context_tokens: int = Field(gt=512, le=32_768)
    max_memory_per_gpu_gib: float = Field(gt=1.0, le=15.0, default=14.0)
    probe_artifact_sha256: str | None = None
    adapter_id: str | None = None
    adapter_revision: str | None = None

    @model_validator(mode="after")
    def _pins_and_adapter(self) -> MatrixModel:
        for name, value in (
            ("revision", self.revision), ("tokenizer_revision", self.tokenizer_revision)
        ):
            if not value or value.startswith("PIN"):
                raise ValueError(f"{name} must be an exact immutable revision")
        if bool(self.adapter_id) != bool(self.adapter_revision):
            raise ValueError("adapter_id and adapter_revision must be set together")
        if self.probe_artifact_sha256 and (
            len(self.probe_artifact_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.probe_artifact_sha256)
        ):
            raise ValueError("probe_artifact_sha256 must be 64 lowercase hex characters")
        return self


class MatrixCell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str
    generation_cohort_id: str
    agent: MatrixModel
    monitor: MatrixModel
    fact_checker: MatrixModel | None = None
    account_slot: Literal[0, 1, 2]
    replicate_ids: list[int] = [0]
    primary: bool = False
    # Freeze-record annotations (provenance for the operator; not scheduling
    # inputs): lifecycle status, the realized cohort run, the derived cell
    # config, and the deterministic suffixed monitor score name.
    status: str | None = None
    cohort_run: str | None = None
    cell_config: str | None = None
    monitor_score_name: str | None = None

    @model_validator(mode="after")
    def _validate_profiles(self) -> MatrixCell:
        for role, spec in (("agent", self.agent), ("monitor", self.monitor)):
            profile = validate_model_assignment(
                spec.model_id,
                role=role,  # type: ignore[arg-type]
                quantization=spec.quantization,
                allow_probe_required=bool(spec.probe_artifact_sha256),
            )
            if profile.provider != spec.provider:
                raise ValueError(f"{spec.model_id}: provider disagrees with audited registry")
            if profile.reasoning_protocol != spec.reasoning_protocol:
                raise ValueError(f"{spec.model_id}: reasoning protocol disagrees with registry")
        if self.fact_checker:
            profile = validate_model_assignment(
                self.fact_checker.model_id,
                role="fact_checker",
                quantization=self.fact_checker.quantization,
                allow_probe_required=bool(self.fact_checker.probe_artifact_sha256),
            )
            monitor_profile = get_model_profile(self.monitor.model_id)
            if (
                profile.organization == monitor_profile.organization
                or profile.architecture_family == monitor_profile.architecture_family
            ):
                raise ValueError("fact checker must differ in organization and architecture family")
        if len(self.replicate_ids) != len(set(self.replicate_ids)):
            raise ValueError("replicate_ids must be unique")
        return self


class ModelMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    study_id: str
    benchmark_config: str
    frozen_code_revision: str
    cells: list[MatrixCell]

    @model_validator(mode="after")
    def _unique_and_anchored(self) -> ModelMatrix:
        ids = [cell.cell_id for cell in self.cells]
        if len(ids) != len(set(ids)):
            raise ValueError("matrix cell_id values must be unique")
        if sum(cell.primary for cell in self.cells) != 1:
            raise ValueError("exactly one primary anchor cell is required")
        if not self.frozen_code_revision or self.frozen_code_revision.startswith("PIN"):
            raise ValueError("frozen_code_revision must be pinned")
        cohorts: dict[str, tuple[dict, int]] = {}
        for cell in self.cells:
            agent = cell.agent.model_dump(mode="json")
            existing = cohorts.setdefault(
                cell.generation_cohort_id, (agent, cell.account_slot)
            )
            if existing != (agent, cell.account_slot):
                raise ValueError(
                    f"generation cohort {cell.generation_cohort_id!r} has multiple agent "
                    "specs or account assignments"
                )
        return self

    def content_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_model_matrix(path: str | Path) -> ModelMatrix:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ModelMatrix.model_validate(payload)


def build_schedule(matrix: ModelMatrix) -> dict[str, Any]:
    monitoring_jobs = []
    generation_jobs_by_id: dict[str, dict[str, Any]] = {}
    for cell in matrix.cells:
        for replicate_id in sorted(cell.replicate_ids):
            generation_job_id = f"{cell.generation_cohort_id}.r{replicate_id}"
            generation_jobs_by_id.setdefault(generation_job_id, {
                "job_id": generation_job_id,
                "generation_cohort_id": cell.generation_cohort_id,
                "replicate_id": replicate_id,
                "account_slot": cell.account_slot,
                "agent_model_id": cell.agent.model_id,
            })
            monitoring_jobs.append({
                "job_id": f"{cell.cell_id}.r{replicate_id}",
                "cell_id": cell.cell_id,
                "generation_job_id": generation_job_id,
                "replicate_id": replicate_id,
                "account_slot": cell.account_slot,
                "primary": cell.primary,
                "agent_model_id": cell.agent.model_id,
                "monitor_model_id": cell.monitor.model_id,
                "fact_checker_model_id": (
                    cell.fact_checker.model_id if cell.fact_checker else None
                ),
            })
    generation_jobs = sorted(
        generation_jobs_by_id.values(), key=lambda item: (item["account_slot"], item["job_id"])
    )
    monitoring_jobs.sort(key=lambda item: (item["account_slot"], item["job_id"]))
    return {
        "schema_version": "1.0.0",
        "study_id": matrix.study_id,
        "matrix_sha256": matrix.content_hash(),
        "frozen_code_revision": matrix.frozen_code_revision,
        "generation_jobs": generation_jobs,
        "monitoring_jobs": monitoring_jobs,
        "jobs_by_account": {
            str(slot): {
                "generation": sum(job["account_slot"] == slot for job in generation_jobs),
                "monitoring": sum(job["account_slot"] == slot for job in monitoring_jobs),
            }
            for slot in range(3)
        },
    }


def _agent_payload(spec: MatrixModel) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "provider": spec.provider,
        "reasoning_protocol": spec.reasoning_protocol,
        "quantization": spec.quantization,
        "compute_dtype": spec.compute_dtype,
        "attention_backend": spec.attention_backend,
        "max_context_tokens": spec.max_context_tokens,
        "max_memory_per_gpu_gib": spec.max_memory_per_gpu_gib,
        "probe_artifact_sha256": spec.probe_artifact_sha256,
        "adapter_id": spec.adapter_id,
        "adapter_revision": spec.adapter_revision,
    }


def _monitor_payload(spec: MatrixModel) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "model_family": get_model_profile(spec.model_id).architecture_family,
        "provider": spec.provider,
        "reasoning_protocol": spec.reasoning_protocol,
        "quantization": spec.quantization,
        "compute_dtype": spec.compute_dtype,
        "attention_backend": spec.attention_backend,
        "max_context_tokens": spec.max_context_tokens,
        "max_memory_per_gpu_gib": spec.max_memory_per_gpu_gib,
        "probe_artifact_sha256": spec.probe_artifact_sha256,
        "adapter_id": spec.adapter_id,
        "adapter_revision": spec.adapter_revision,
    }


def materialize_matrix(matrix: ModelMatrix, output_dir: str | Path) -> dict[str, Any]:
    base_path = Path(matrix.benchmark_config)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    destination = Path(output_dir)
    schedule = build_schedule(matrix)
    cells = {cell.cell_id: cell for cell in matrix.cells}
    cohort_cell: dict[str, MatrixCell] = {}
    for cell in matrix.cells:
        cohort_cell.setdefault(cell.generation_cohort_id, cell)

    for job in schedule["generation_jobs"]:
        cell = cohort_cell[job["generation_cohort_id"]]
        payload = yaml.safe_load(yaml.safe_dump(base))
        payload["agent"].update(_agent_payload(cell.agent))
        payload["monitor"].update(_monitor_payload(cell.monitor))
        payload["provenance"].update({
            "code_revision": matrix.frozen_code_revision,
            "experiment_name": f"{matrix.study_id}:{cell.generation_cohort_id}",
            "matrix_cell_id": f"generation:{cell.generation_cohort_id}",
            "replicate_id": job["replicate_id"],
        })
        path = destination / "generation" / f"{job['job_id']}.yaml"
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=True))
        job["config_path"] = str(path)

    for job in schedule["monitoring_jobs"]:
        cell = cells[job["cell_id"]]
        payload = yaml.safe_load(yaml.safe_dump(base))
        payload["agent"].update(_agent_payload(cell.agent))
        payload["monitor"].update(_monitor_payload(cell.monitor))
        if cell.fact_checker:
            checker = cell.fact_checker
            payload["monitor"].update({
                "fact_checker_model_id": checker.model_id,
                "fact_checker_model_revision": checker.revision,
                "fact_checker_model_family": get_model_profile(
                    checker.model_id
                ).architecture_family,
                "fact_checker_provider": checker.provider,
                "fact_checker_reasoning_protocol": checker.reasoning_protocol,
                "fact_checker_quantization": checker.quantization,
                "fact_checker_compute_dtype": checker.compute_dtype,
                "fact_checker_attention_backend": checker.attention_backend,
                "fact_checker_max_context_tokens": checker.max_context_tokens,
                "fact_checker_max_memory_per_gpu_gib": checker.max_memory_per_gpu_gib,
                "fact_checker_adapter_id": checker.adapter_id,
                "fact_checker_adapter_revision": checker.adapter_revision,
                "fact_checker_probe_artifact_sha256": checker.probe_artifact_sha256,
                "require_cross_family_fact_checker": True,
            })
        payload["provenance"].update({
            "code_revision": matrix.frozen_code_revision,
            "experiment_name": f"{matrix.study_id}:{cell.generation_cohort_id}",
            "matrix_cell_id": cell.cell_id,
            "replicate_id": job["replicate_id"],
        })
        path = destination / "monitoring" / f"{job['job_id']}.yaml"
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=True))
        job["config_path"] = str(path)
    atomic_write_json(destination / "schedule.json", schedule)
    return schedule


__all__ = [
    "ModelMatrix", "build_schedule", "load_model_matrix", "materialize_matrix"
]
