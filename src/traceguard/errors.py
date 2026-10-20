"""Stable, serializable failure taxonomy for unattended research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    CONFIGURATION = "configuration"
    INPUT_INTEGRITY = "input_integrity"
    MODEL_ADMISSION = "model_admission"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TRANSIENT_PROVIDER = "transient_provider"
    DATA_QUALITY = "data_quality"
    STATISTICAL_GATE = "statistical_gate"
    SOFTWARE_DEFECT = "software_defect"
    USER_CANCELLED = "user_cancelled"
    UNKNOWN = "unknown"


RETRYABLE_CODES = {ErrorCode.RESOURCE_EXHAUSTED, ErrorCode.TRANSIENT_PROVIDER}


@dataclass(frozen=True)
class FailureEnvelope:
    code: ErrorCode
    message: str
    stage: str
    exception_type: str | None = None
    retryable: bool = False
    context: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "code": self.code.value,
            "message": self.message,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "retryable": self.retryable,
            "context": self.context or {},
        }


def classify_exception(exc: BaseException, stage: str) -> FailureEnvelope:
    text = str(exc).lower()
    if isinstance(exc, (FileNotFoundError, ValueError, TypeError)):
        code = ErrorCode.CONFIGURATION
    elif "out of memory" in text or "cuda oom" in text:
        code = ErrorCode.RESOURCE_EXHAUSTED
    elif "timeout" in text or "rate limit" in text or "temporar" in text:
        code = ErrorCode.TRANSIENT_PROVIDER
    else:
        code = ErrorCode.UNKNOWN
    return FailureEnvelope(
        code=code,
        message=str(exc),
        stage=stage,
        exception_type=type(exc).__name__,
        retryable=code in RETRYABLE_CODES,
    )


__all__ = ["ErrorCode", "FailureEnvelope", "classify_exception"]
