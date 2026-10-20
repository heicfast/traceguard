"""Condition harness: compile all monitor views for a set of episodes.

Pairing discipline (docs/01 §6 and §8):
* All offline conditions score the SAME stored episodes.
* Condition E (shuffled CoT) uses, for each episode, the reasoning trace of
  its deterministic partner: the episode with the same template, instance,
  and generation seed but the other policy mode. The rule is label-blind in
  the sense that it is fixed and symmetric; it never depends on outcomes.
  Partner traces never cross split-role boundaries (same family => same
  split role by construction).
* Views are compiled through ``compile_view_checked`` so a leakage
  assertion runs before any monitor sees the text.
"""

from __future__ import annotations

from typing import Any

from ..tasks.specs import TaskInstance, instantiate
from ..traces.views import MonitorObservation, compile_view_checked


def episode_partner_key(envelope: dict[str, Any]) -> tuple[str, int, int]:
    prov = envelope["provenance"]
    return (prov["template_name"], prov["instance_index"], envelope["seed"])


def build_partner_index(
        envelopes: list[dict[str, Any]]) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    """Map (partner_key, policy_mode) -> envelope for the OPPOSITE-mode lookup.

    Keying by policy mode is what makes condition E symmetric: an episode's
    shuffle partner is BY DEFINITION its policy twin, and the lookup can never
    return the episode itself. (Review finding 2026-10-19: the previous
    last-write-wins index without the policy component silently produced
    partner=None for one twin, making attack-side E degenerate to condition A.)
    """
    index: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for env in envelopes:
        index[episode_partner_key(env) + (env["policy_mode"],)] = env
    return index


def find_partner(index: dict[tuple[str, int, int, str], dict[str, Any]],
                 envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Return the policy twin of ``envelope`` from the index (never itself)."""
    opposite = "attack" if envelope["policy_mode"] == "honest" else "honest"
    return index.get(episode_partner_key(envelope) + (opposite,))


def load_task_instance_for(envelope: dict[str, Any]) -> TaskInstance:
    prov = envelope["provenance"]
    return instantiate(prov["template_name"], prov["instance_seed"], prov["control_variant"])


def compile_all_conditions(
    envelope: dict[str, Any],
    conditions: list[str],
    partner_envelope: dict[str, Any] | None,
    hidden_instruction: str | None,
    *,
    partner_hidden_instruction: str | None = None,
    cutoff_event_index: int | None = None,
    reasoning_window_chars: int | None = None,
) -> dict[str, MonitorObservation]:
    """Compile + cleanliness-check every requested condition for one episode."""
    task = load_task_instance_for(envelope)
    observations: dict[str, MonitorObservation] = {}
    for condition in conditions:
        kwargs: dict[str, Any] = {}
        if condition == "actions_plus_shuffled_cot":
            kwargs["shuffle_source_envelope"] = partner_envelope
        obs = compile_view_checked(
            envelope,
            task_authorization=task.user_goal,
            public_policy=task.public_policy_text(),
            condition=condition,
            hidden_instruction=hidden_instruction,
            partner_hidden_instruction=partner_hidden_instruction,
            cutoff_event_index=cutoff_event_index,
            reasoning_window_chars=reasoning_window_chars,
            **kwargs,
        )
        observations[condition] = obs
    return observations


__all__ = [
    "compile_all_conditions", "build_partner_index", "episode_partner_key",
    "find_partner", "load_task_instance_for",
]
