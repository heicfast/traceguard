"""Pure, typed sandbox state.

The state is a plain JSON-able structure so it can be hashed canonically,
serialized into episode evidence, and replayed deterministically. State is
immutable by convention: :func:`sandbox.engine.step` always returns a new
state; callers never mutate in place.
"""

from __future__ import annotations

from typing import Any

from ..ids import sha256_json

STATE_SCHEMA_VERSION = "1.0.0"


class SandboxState:
    """Immutable, hashable snapshot of the simulated environment.

    ``objects`` maps whitelisted object paths to nested JSON values.
    ``step_count`` increments on every accepted tool transition.
    """

    __slots__ = ("objects", "step_count", "submitted")

    def __init__(self, objects: dict[str, Any], step_count: int = 0, submitted: bool = False):
        self.objects = objects
        self.step_count = step_count
        self.submitted = submitted

    # -- serialization -----------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        return {
            "objects": self.objects,
            "step_count": self.step_count,
            "submitted": self.submitted,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SandboxState:
        cls._validate_payload(payload)
        return cls(objects=payload["objects"], step_count=int(payload["step_count"]),
                   submitted=bool(payload["submitted"]))

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or set(payload) != {"objects", "step_count", "submitted"}:
            raise ValueError("state payload must have exactly keys objects, step_count, submitted")
        if not isinstance(payload["objects"], dict):
            raise ValueError("state.objects must be a dict of path -> object")
        for k, v in payload["objects"].items():
            if not isinstance(k, str) or not k:
                raise ValueError("object paths must be non-empty strings")
            if not isinstance(v, (dict, list)):
                raise ValueError(f"object at {k!r} must be a JSON object or list")

    # -- hashing -----------------------------------------------------------
    def hash(self) -> str:
        return sha256_json({"schema": STATE_SCHEMA_VERSION, **self.to_payload()})

    # -- equality / debugging -----------------------------------------------
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SandboxState) and self.to_payload() == other.to_payload()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SandboxState(paths={sorted(self.objects)}, step_count={self.step_count}, submitted={self.submitted})"


def deep_copy_objects(objects: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy JSON-able object storage (dicts/lists/str/num/bool/None only)."""
    import copy
    return copy.deepcopy(objects)


def _index_or_key(current: Any, key: str) -> tuple[bool, Any]:
    """Navigate one level: dict key lookup, or integer index into a list."""
    if isinstance(current, dict):
        if key in current:
            return True, current[key]
        return False, None
    if isinstance(current, list):
        try:
            ix = int(key)
        except ValueError:
            return False, None
        if 0 <= ix < len(current):
            return True, current[ix]
        return False, None
    return False, None


def get_at_path(obj: Any, key_path: list[str]) -> tuple[bool, Any]:
    """Navigate a nested dict/list by key path. Returns (found, value).

    Integer-looking keys index into JSON arrays (e.g. ['events', '0']).
    """
    current = obj
    for key in key_path:
        found, current = _index_or_key(current, key)
        if not found:
            return False, None
    return True, current


def set_at_path(obj: dict, key_path: list[str], value: Any) -> None:
    """Set value at nested key path, creating intermediate dicts as needed.

    Integer-looking keys index into existing lists (must exist; never grown).
    """
    if not key_path:
        raise ValueError("key_path must be non-empty")
    current: Any = obj
    for key in key_path[:-1]:
        if isinstance(current, dict):
            nxt = current.get(key)
            if not isinstance(nxt, (dict, list)):
                nxt = {}
                current[key] = nxt
            current = nxt
        elif isinstance(current, list):
            current = current[int(key)]  # raises IndexError on bad path
        else:
            raise TypeError(f"cannot navigate into {type(current).__name__}")
    last = key_path[-1]
    if isinstance(current, dict):
        current[last] = value
    elif isinstance(current, list):
        current[int(last)] = value
    else:
        raise TypeError(f"cannot assign into {type(current).__name__}")


def delete_at_path(obj: dict, key_path: list[str]) -> bool:
    """Delete key at nested path (dict key or list index). Returns True if removed."""
    if not key_path:
        return False
    current: Any = obj
    for key in key_path[:-1]:
        found, current = _index_or_key(current, key)
        if not found:
            return False
    last = key_path[-1]
    if isinstance(current, dict) and last in current:
        del current[last]
        return True
    if isinstance(current, list):
        try:
            ix = int(last)
        except ValueError:
            return False
        if 0 <= ix < len(current):
            del current[ix]
            return True
        return False
    return False
