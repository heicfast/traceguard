"""Stable identifiers, canonical JSON, and SHA-256 hashing.

Design rules (from docs/02_ARCHITECTURE.md §4 and §5):

* Canonical JSON serialization fixes key ordering and delimiters so that the
  same logical object always hashes to the same value.
* Episode identifiers are OPAQUE. They must never encode ``policy_mode``,
  attack success, or any other privileged label, because identifiers can
  leak labels into monitor-visible contexts (the "filename-as-label" leak).
* SHA-256 is used for accidental-corruption checks and deterministic
  identification, not as confidentiality protection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize to a deterministic, hashable JSON string.

    Key ordering is fixed, separators are compact, and non-ASCII is escaped so
    that the byte representation depends only on the logical content.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string, hex encoded."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON serialization of ``obj``."""
    return sha256_text(canonical_json(obj))


def short_hash(obj: Any, length: int = 16) -> str:
    """Short (default 64-bit) prefix of the canonical-JSON hash."""
    digest = sha256_json(obj)
    if length > len(digest):
        raise ValueError("length exceeds digest size")
    return digest[:length]


def episode_id(run_id: str, task_instance_id: str, policy_mode: str, seed: int) -> str:
    """Opaque episode identifier.

    Contains a hash over the identity tuple but NO readable label: an episode id
    must not reveal whether the episode is honest or attacked. The mapping from
    id to privileged fields is stored only in the privileged record store.
    """
    if policy_mode not in ("honest", "attack"):
        raise ValueError(f"invalid policy_mode: {policy_mode!r}")
    core = sha256_json({"run_id": run_id, "task_instance_id": task_instance_id,
                         "policy_mode": policy_mode, "seed": seed})
    return f"ep-{core[:24]}"


def task_instance_id(template_id: str, instance_seed: int) -> str:
    """Opaque identifier for one seeded parameterization of a task template."""
    core = sha256_json({"template_id": template_id, "instance_seed": instance_seed})
    return f"ti-{core[:24]}"


def task_template_id(family_id: str, template_name: str, revision: int) -> str:
    """Readable-but-unlabeled template identifier (template content is public)."""
    return f"tt-{family_id}-{template_name}-r{revision}"


def monitor_score_id(episode_id: str, condition: str, monitor_name: str, prompt_hash: str) -> str:
    """Opaque identifier for one monitor-scoring event."""
    core = sha256_json({"episode_id": episode_id, "condition": condition,
                         "monitor_name": monitor_name, "prompt_hash": prompt_hash})
    return f"ms-{core[:24]}"


def is_opaque_identifier(candidate: str) -> bool:
    """Check that an identifier string does not embed privileged vocabulary.

    Any occurrence of attack/honest/success vocabulary inside an identifier is
    a label-leak hazard. Used by leakage tests.
    """
    forbidden = ("attack", "honest", "violation", "succeed", "success", "unauthorized",
                 "policy_mode", "label", "positive", "negative")
    lowered = candidate.lower()
    return not any(word in lowered for word in forbidden)
