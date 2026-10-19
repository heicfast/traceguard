"""Typed access to episode envelopes + schema validation.

Loads/validates privileged episode envelopes against the pinned JSON Schema
and provides typed accessors. Validation failures are fatal: a record that
fails validation must never silently enter analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
_SCHEMA_PATH = _SCHEMA_DIR / "episode.schema.json"
_schema_cache: dict[str, dict] = {}


def episode_schema() -> dict[str, Any]:
    raw = _SCHEMA_PATH.read_text(encoding="utf-8")
    key = _SCHEMA_PATH.name
    if key not in _schema_cache:
        _schema_cache[key] = json.loads(raw)
    return _schema_cache[key]


def validate_envelope(envelope: dict[str, Any]) -> None:
    """Strict schema validation; raises jsonschema.ValidationError on failure."""
    jsonschema.Draft202012Validator.check_schema(episode_schema())
    validator = jsonschema.Draft202012Validator(episode_schema())
    errors = sorted(validator.iter_errors(envelope), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise jsonschema.ValidationError(
            f"envelope schema violation at {list(first.path)}: {first.message}"
        )


def _validate_named(payload: dict[str, Any], filename: str, label: str) -> None:
    path = _SCHEMA_DIR / filename
    if filename not in _schema_cache:
        _schema_cache[filename] = json.loads(path.read_text(encoding="utf-8"))
    schema = _schema_cache[filename]
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(payload), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise jsonschema.ValidationError(
            f"{label} schema violation at {list(first.path)}: {first.message}"
        )


def validate_privileged(payload: dict[str, Any]) -> None:
    _validate_named(payload, "privileged.schema.json", "privileged record")


def validate_monitor_score(payload: dict[str, Any]) -> None:
    _validate_named(payload, "monitor_score.schema.json", "monitor score")


def load_episodes(path: str | Path, validate: bool = True) -> list[dict[str, Any]]:
    """Load a JSONL shard of episode envelopes."""
    records: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        if validate:
            try:
                validate_envelope(rec)
            except jsonschema.ValidationError as exc:
                raise ValueError(f"{path}:{line_no} fails schema: {exc.message}") from exc
        records.append(rec)
    return records


def iter_event_types(envelope: dict[str, Any]) -> list[str]:
    return [e["type"] for e in envelope["events"]]


__all__ = [
    "episode_schema",
    "validate_envelope",
    "validate_privileged",
    "validate_monitor_score",
    "load_episodes",
    "iter_event_types",
]
