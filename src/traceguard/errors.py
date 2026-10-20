"""Stable, serializable failure taxonomy for unattended research workflows.

Exit-code convention (docs/AUTOMATION_AND_EVIDENCE_V6.md §3):

* ``0``  success.
* ``1``  generic failure.
* ``2``  CLI usage error (argparse).
* ``3``  statistical-gate failure: a predeclared threshold was not met. Stage
  commands that evaluate predeclared scientific gates (``scripts/pilot_gate.py``,
  ``scripts/monitor_gate.py``) exit ``GATE_FAILURE_EXIT_CODE`` on failure, and
  the workflow engine classifies such exits as ``statistical_gate``: a
  non-retryable class that must never be retried into a pass. The traceguard
  CLI in turn exits 3 when a workflow stops on a statistical-gate failure.

Retryability policy (two layers, applied by the workflow engine):

1. Class-intrinsic retryability. ``RETRYABLE_CODES`` lists failure classes
   that are inherently retryable (``resource_exhausted``, ``transient_provider``).
   Engine-detected exceptions classified by :func:`classify_exception` are
   retried only when their class is in this set and the stage still has
   declared attempts left. All other engine-detected classes
   (``configuration``, ``unknown``, ...) stop the stage immediately:
   deterministic misconfiguration cannot be retried into success.
2. Operator-declared retries. A stage command that exits nonzero (without the
   gate exit code) is a verdict from the stage itself; the engine cannot see
   whether the cause was transient, so the per-stage ``retries`` field in the
   workflow file governs and the failure is classified ``software_defect``.

``statistical_gate`` and ``user_cancelled`` are absolute non-retry classes in
every layer: gate failures are never retried into a pass and an interrupted
orchestrator is never silently resumed as if nothing happened.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final


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

#: Failure classes that must never be retried in any layer (see module docstring).
NEVER_RETRY_CODES = {ErrorCode.STATISTICAL_GATE, ErrorCode.USER_CANCELLED}

#: Distinct CLI exit code for statistical-gate failures (see module docstring).
GATE_FAILURE_EXIT_CODE: Final[int] = 3


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
    if isinstance(exc, subprocess.TimeoutExpired):
        code = ErrorCode.TRANSIENT_PROVIDER
    elif isinstance(exc, (FileNotFoundError, ValueError, TypeError)):
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


def classify_exit_code(returncode: int) -> ErrorCode | None:
    """Map a stage-command exit code onto the failure taxonomy.

    Returns ``STATISTICAL_GATE`` for the predeclared gate exit code, and
    ``None`` for exit codes with no taxonomy meaning (generic failure).
    """
    if returncode == GATE_FAILURE_EXIT_CODE:
        return ErrorCode.STATISTICAL_GATE
    return None


def exit_code_for_error(code: ErrorCode) -> int:
    """Reverse mapping: the CLI exit code that surfaces a failure class."""
    if code == ErrorCode.STATISTICAL_GATE:
        return GATE_FAILURE_EXIT_CODE
    return 1


def retry_decision(code: ErrorCode, *, attempts_remaining: bool) -> bool:
    """Final retry decision for a failure class under the two-layer policy."""
    if code in NEVER_RETRY_CODES:
        return False
    if code in RETRYABLE_CODES:
        return attempts_remaining
    if code is ErrorCode.SOFTWARE_DEFECT:
        # Stage-exit verdict: the operator's declared retries govern.
        return attempts_remaining
    # Engine-detected non-retryable classes (configuration, unknown, ...).
    return False


__all__ = [
    "GATE_FAILURE_EXIT_CODE",
    "NEVER_RETRY_CODES",
    "ErrorCode",
    "FailureEnvelope",
    "classify_exception",
    "classify_exit_code",
    "exit_code_for_error",
    "retry_decision",
]
