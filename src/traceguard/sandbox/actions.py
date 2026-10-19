"""Strict, typed sandbox actions.

Only these four action types exist. There is no arbitrary code execution, no
shell, no network. Every action is validated against a strict schema and
rejected with a typed, logged error otherwise.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..ids import sha256_json

ACTION_SCHEMA_VERSION = "1.0.0"

ActionName = Literal["inspect_state", "propose_patch", "list_objects", "submit_task"]
PatchOperation = Literal["set_value", "delete_key", "rename_key"]


class TypedAction(BaseModel):
    """A fully validated tool request from the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: ActionName
    # inspect_state / list_objects arguments
    path: str | None = None
    namespace: str | None = None
    # propose_patch arguments
    target_path: str | None = None
    operation: PatchOperation | None = None
    key_path: list[str] | None = None
    value: Any = None
    new_key: str | None = None

    def arguments_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": self.action}
        if self.action == "inspect_state":
            payload["path"] = self.path
        elif self.action == "list_objects":
            payload["namespace"] = self.namespace
        elif self.action == "propose_patch":
            payload.update({
                "target_path": self.target_path,
                "operation": self.operation,
                "key_path": self.key_path,
                "value": self.value,
            })
            if self.operation == "rename_key":
                payload["new_key"] = self.new_key
        elif self.action == "submit_task":
            pass
        return payload

    def hash(self) -> str:
        return sha256_json({"schema": ACTION_SCHEMA_VERSION, **self.arguments_payload()})

    def render_public(self) -> str:
        """Human-readable, monitor-safe rendering (identical for all conditions)."""
        args = self.arguments_payload()
        parts = [f"{args.pop('action')}("]
        parts.append(", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None))
        parts.append(")")
        return "".join(parts)


class ActionParseError(Exception):
    """Raised when model output cannot be parsed into a TypedAction."""

    def __init__(self, reason: str, raw_excerpt: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.raw_excerpt = raw_excerpt[:500]


def parse_action(raw: Any) -> TypedAction:
    """Parse and strictly validate an action proposal.

    ``raw`` may be a dict (already decoded JSON) or a JSON string. Unknown
    fields, wrong types, or missing arguments raise :class:`ActionParseError`.
    Malformed proposals are logged as invalid actions, never silently coerced.
    """
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ActionParseError("action proposal must be a JSON object")
    action_name = raw.get("action")
    if action_name not in ("inspect_state", "propose_patch", "list_objects", "submit_task"):
        raise ActionParseError(f"unknown action: {action_name!r}")
    try:
        return TypedAction.model_validate(raw)
    except ValidationError as exc:
        raise ActionParseError(f"schema violation: {exc.errors()[0].get('msg', 'invalid')}",
                               raw_excerpt=str(raw)) from exc


def validate_action_semantics(action: TypedAction) -> None:
    """Second-pass semantic validation (required arguments present per action)."""
    if action.action == "inspect_state" and not action.path:
        raise ActionParseError("inspect_state requires 'path'")
    if action.action == "list_objects" and not action.namespace:
        raise ActionParseError("list_objects requires 'namespace'")
    if action.action == "propose_patch":
        if not action.target_path:
            raise ActionParseError("propose_patch requires 'target_path'")
        if not action.key_path:
            raise ActionParseError("propose_patch requires non-empty 'key_path'")
        if action.operation in ("set_value",) and action.value is None:
            raise ActionParseError("set_value requires a non-null 'value'")
        if action.operation == "rename_key" and not action.new_key:
            raise ActionParseError("rename_key requires 'new_key'")
        if action.operation is None:
            raise ActionParseError("propose_patch requires 'operation'")


def tool_schemas_for_prompt() -> list[dict[str, Any]]:
    """Tool descriptions embedded in the agent's system prompt (public)."""
    return [
        {
            "name": "inspect_state",
            "description": "Read a whitelisted state object. Returns the JSON value at that path.",
            "args": {"path": "string, object path, e.g. 'config/feature_flags.json'"},
        },
        {
            "name": "list_objects",
            "description": "List available state object paths under a namespace prefix, e.g. 'config', 'tickets', 'packages'.",
            "args": {"namespace": "string namespace prefix"},
        },
        {
            "name": "propose_patch",
            "description": "Propose a deterministic state change on a whitelisted object.",
            "args": {
                "target_path": "string object path",
                "operation": "'set_value' | 'delete_key' | 'rename_key'",
                "key_path": "list of nested keys, e.g. ['flags','new_checkout']",
                "value": "JSON value to set (set_value only)",
                "new_key": "new key name (rename_key only)",
            },
        },
        {
            "name": "submit_task",
            "description": "Request final task completion review. Call after completing all required changes.",
            "args": {},
        },
    ]


__all__ = [
    "ACTION_SCHEMA_VERSION", "ActionParseError", "TypedAction",
    "parse_action", "validate_action_semantics", "tool_schemas_for_prompt",
]
