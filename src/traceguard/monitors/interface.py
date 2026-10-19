"""Monitor contract and score records.

A Monitor sees ONLY compiled MonitorObservations. Scores are structured:
suspicion in [0,1] plus validity metadata. Invalid scores are recorded as
invalid and NEVER silently imputed or retried beyond the preregistered policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..ids import sha256_text
from ..traces.views import MonitorObservation


@dataclass
class MonitorScore:
    """One scoring event, fully auditable."""

    episode_id: str
    condition: str
    monitor_name: str
    suspicion: float
    valid: bool
    invalid_reason: str | None = None
    prompt_hash: str = ""
    model_revision: str = ""
    input_token_estimate: int = 0
    output_token_count: int = 0
    elapsed_ms: float = 0.0
    quoted_evidence_ids: list[str] = field(default_factory=list)
    view_hash: str = ""
    score_id: str = ""
    prompt_template_hash: str = ""
    rendered_prompt_hash: str = ""
    rendered_prompt: str = ""
    raw_response: str = ""
    parsed_response: dict[str, Any] | None = None
    verdict_text: str = ""
    inference_audit: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> MonitorScore:
        if not self.score_id:
            from ..ids import monitor_score_id
            self.score_id = monitor_score_id(self.episode_id, self.condition,
                                             self.monitor_name, self.prompt_hash)
        if self.view_hash:
            self.view_hash = self.view_hash
        return self

    def to_payload(self) -> dict[str, Any]:
        return {
            "record_version": "2.0.0",
            "score_id": self.score_id,
            "episode_id": self.episode_id,
            "condition": self.condition,
            "monitor_name": self.monitor_name,
            "suspicion": self.suspicion,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "prompt_hash": self.prompt_hash,
            "model_revision": self.model_revision,
            "input_token_estimate": self.input_token_estimate,
            "output_token_count": self.output_token_count,
            "elapsed_ms": self.elapsed_ms,
            "quoted_evidence_ids": self.quoted_evidence_ids,
            "view_hash": self.view_hash,
            "prompt_template_hash": self.prompt_template_hash,
            "rendered_prompt_hash": self.rendered_prompt_hash,
            "rendered_prompt": self.rendered_prompt,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "verdict_text": self.verdict_text,
            "inference_audit": self.inference_audit,
        }


class Monitor(Protocol):
    """score(observation) -> MonitorScore; deterministic given the view."""

    name: str

    def score(self, observation: MonitorObservation) -> MonitorScore: ...


def prompt_hash(text: str) -> str:
    return sha256_text(text)


__all__ = ["Monitor", "MonitorScore", "prompt_hash"]
