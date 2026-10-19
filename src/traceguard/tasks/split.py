"""Family-scoped split manifest (frozen).

Assignment is at the FAMILY level (group-based, per docs/01 §5): every
template of a family belongs to the same split, so the final holdout and the
shift set are genuine distribution shifts (unseen families, unseen tools
vocabularies, unseen protected-field shapes) rather than near-duplicates of
development tasks.

Roles:
  development  - monitor prompt iteration and qualitative debugging
  validation   - threshold calibration and condition ablations
  holdout      - final confirmatory evaluation (labels sealed until analysis)
  shift        - held-out environment families for RQ2 robustness analysis

The manifest hash pins the assignment; any change requires a new benchmark
version and a dated deviation note.
"""

from __future__ import annotations

from ..ids import sha256_json

FAMILY_SPLIT: dict[str, str] = {
    "config_update": "development",
    "data_migration": "development",
    "dependency_metadata": "validation",
    "incident_workflow": "holdout",
    "service_rotation": "shift",
    "backup_policy": "shift",
}

ROLES = ("development", "validation", "holdout", "shift")


def manifest_payload() -> dict[str, str]:
    return dict(FAMILY_SPLIT)


def manifest_hash() -> str:
    return sha256_json({"split_manifest": FAMILY_SPLIT, "version": "1.0.0"})


def role_for_family(family_id: str) -> str:
    try:
        return FAMILY_SPLIT[family_id]
    except KeyError as exc:
        raise KeyError(f"family {family_id!r} is not in the frozen split manifest") from exc


def families_for_role(role: str) -> list[str]:
    if role not in ROLES:
        raise ValueError(f"unknown split role: {role!r}")
    return [f for f, r in FAMILY_SPLIT.items() if r == role]


def assert_split_consistent(family_ids: list[str]) -> None:
    """Every family used in an experiment must appear in the frozen manifest."""
    missing = [f for f in family_ids if f not in FAMILY_SPLIT]
    if missing:
        raise ValueError(f"families missing from split manifest: {missing}")


__all__ = [
    "FAMILY_SPLIT", "ROLES", "manifest_payload", "manifest_hash",
    "role_for_family", "families_for_role", "assert_split_consistent",
]
