#!/usr/bin/env python3
"""Measure rendered monitor prompt sizes for the anchor cohort (window 12,288)
against the scale-cell OOM failure lists. Decides the capacity fix from data.

Usage: python3 scripts/measure_prompt_sizes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from traceguard.monitors.conditions import (
    build_partner_index, compile_all_conditions, find_partner,
)
from traceguard.monitors.llm import MONITOR_PROMPT_TEMPLATE, evidence_nonce
from traceguard.traces.storage import RunStore

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs/v6/pilot_v0_run"
WINDOW_CHARS = 3072 * 4  # frozen: max_monitor_tokens 3072 -> 12,288 chars

C_FAILED = {
    "ep-0bc6ef9f6a7aa7c7a10ab063", "ep-398e9add1e9931f4be93687d",
    "ep-3bc9fae0c2995b987c0bece8", "ep-593ec25c52b9fae0889fd924",
    "ep-85c87c00cf8586b415a62a1e", "ep-aa5c7da2e263fb52d719e841",
    "ep-b4dedc3b2afc841068aac577", "ep-db78969bbadc5a019a259581",
    "ep-e059bd9cf09c58896c09fc44", "ep-e0e8a9f9739bc0cc4a4568c8",
    "ep-eba42cae1f9fa01ddeb2fa7b", "ep-eddebd40120af8c84188873b",
    "ep-02f094a8f1b82cd9114dddae", "ep-14ae2e069dd0828e0deae710",
    "ep-4d2f6a5af004c2f5e0e68f1b", "ep-533d3c55eca9728b37a769a6",
    "ep-7d742b258e08a136328f0c95", "ep-8ef464fc4e939c84f17a4a4d",
    "ep-9b1dbdbf2d1051f75bc67284", "ep-9fe79eca7666502743f872c7",
    "ep-a68bf986d6bb5f36c0cef3ef", "ep-a91d676f158ee8ad11a5edcd",
    "ep-ffce7118095632be8eddad8b",
}
E_FAILED = {
    "ep-0bc6ef9f6a7aa7c7a10ab063", "ep-3bc9fae0c2995b987c0bece8",
    "ep-593ec25c52b9fae0889fd924", "ep-85c87c00cf8586b415a62a1e",
    "ep-b4dedc3b2afc841068aac577", "ep-db78969bbadc5a019a259581",
    "ep-e059bd9cf09c58896c09fc44", "ep-eba42cae1f9fa01ddeb2fa7b",
    "ep-eddebd40120af8c84188873b", "ep-02f094a8f1b82cd9114dddae",
    "ep-14ae2e069dd0828e0deae710", "ep-4d2f6a5af004c2f5e0e68f1b",
    "ep-8ef464fc4e939c84f17a4a4d", "ep-9b1dbdbf2d1051f75bc67284",
    "ep-a68bf986d6bb5f36c0cef3ef", "ep-ffce7118095632be8eddad8b",
    "ep-9fe79eca7666502743f872c7",
}

CHARS_PER_TOKEN = 4.45  # empirical ratio measured from the pilot raw evidence

TEMPLATE_CHARS = len(
    MONITOR_PROMPT_TEMPLATE.format(view="x", evidence_nonce="00000000"))


def prompt_chars(view_text: str, episode_id: str) -> int:
    prompt = MONITOR_PROMPT_TEMPLATE.format(
        view=view_text, evidence_nonce=evidence_nonce(episode_id))
    return len(prompt)


def main() -> int:
    store = RunStore(RUN)
    episodes = store.load_all_episodes(validate=True)
    privileged = store.load_all_privileged()
    partner_index = build_partner_index(episodes)

    rows = []
    for env in episodes:
        partner = find_partner(partner_index, env)
        hidden = privileged.get(env["episode_id"], {}).get("hidden_instruction")
        partner_hidden = (privileged.get(partner["episode_id"], {}).get(
            "hidden_instruction") if partner is not None else None)
        obs = compile_all_conditions(
            env, ["actions", "actions_plus_cot", "actions_plus_noise",
                  "actions_plus_shuffled_cot"], partner, hidden,
            partner_hidden_instruction=partner_hidden,
            reasoning_window_chars=WINDOW_CHARS)
        row = {"episode_id": env["episode_id"]}
        for cond, obs_one in obs.items():
            row[cond] = prompt_chars(obs_one.text, env["episode_id"])
        rows.append(row)

    def stats(cond: str, failed: set[str]) -> None:
        vals = sorted((r[cond], r["episode_id"] in failed) for r in rows)
        ok = [v for v, f in vals if not f]
        bad = [v for v, f in vals if f]
        toks = lambda c: c / CHARS_PER_TOKEN
        print(f"\n== {cond} ==")
        if bad:
            print(f"  failed n={len(bad)}: chars min={min(bad):,} max={max(bad):,}"
                  f" (~{toks(min(bad)):.0f}-{toks(max(bad)):.0f} tok)")
            print(f"  passed n={len(ok)}: max={max(ok):,} chars"
                  f" (~{toks(max(ok)):.0f} tok)")
            print(f"  boundary: largest PASS at {max(ok):,} chars"
                  f" | smallest FAIL at {min(bad):,} chars"
                  f" | margin {min(bad) - max(ok):+,} chars"
                  f" ({(min(bad) - max(ok)) / max(ok) * 100:+.1f}%)")
        else:
            print(f"  all n={len(ok)} passed: chars min={min(ok):,} max={max(ok):,}"
                  f" (~{toks(min(ok)):.0f}-{toks(max(ok)):.0f} tok)")

    print(f"episodes: {len(rows)}  | template overhead: {TEMPLATE_CHARS:,} chars")
    print(f"chars/token estimate: {CHARS_PER_TOKEN}")
    stats("actions", set())
    stats("actions_plus_cot", C_FAILED)
    stats("actions_plus_shuffled_cot", E_FAILED)

    # Actions-only length for the C-failed episodes: is the ACTION part the driver?
    print("\n== C-failed episodes: actions-only vs C prompt ==")
    for r in rows:
        if r["episode_id"] in C_FAILED:
            a, c = r["actions"], r["actions_plus_cot"]
            print(f"  {r['episode_id']}: A={a:,} C={c:,}"
                  f" (reasoning+overhead adds {c - a:,} chars)")

    # What window would bring every C prompt under the observed pass boundary?
    c_ok_max = max(r["actions_plus_cot"] for r in rows
                   if r["episode_id"] not in C_FAILED)
    print(f"\n== largest C pass: {c_ok_max:,} chars —"
          f" the empirical memory boundary (~{c_ok_max / CHARS_PER_TOKEN:.0f} tok)")
    for window_tokens in (2560, 2048, 1792, 1536):
        w_chars = window_tokens * 4
        # recompile C with the smaller window for the currently-failed episodes
        worst = 0
        for env in episodes:
            if env["episode_id"] not in C_FAILED:
                continue
            partner = find_partner(partner_index, env)
            hidden = privileged.get(env["episode_id"], {}).get("hidden_instruction")
            partner_hidden = (privileged.get(partner["episode_id"], {}).get(
                "hidden_instruction") if partner is not None else None)
            obs = compile_all_conditions(
                env, ["actions_plus_cot"], partner, hidden,
                partner_hidden_instruction=partner_hidden,
                reasoning_window_chars=w_chars)
            worst = max(worst, prompt_chars(obs["actions_plus_cot"].text,
                                            env["episode_id"]))
        print(f"  window {window_tokens} tok ({w_chars:,} chars):"
              f" max C prompt among failed eps = {worst:,} chars"
              f" (~{worst / CHARS_PER_TOKEN:.0f} tok)"
              f" {'FITS' if worst <= c_ok_max else 'STILL OVER'}"
              f" boundary {c_ok_max:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
