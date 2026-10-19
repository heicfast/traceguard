#!/usr/bin/env python3
"""Episode-level audit for a TRACEGUARD run (S2 gate discipline).

Reads every episode in a run directory, prints a per-episode summary
(policy, outcome, CoT stats, action funnel) and an aggregate capability
funnel. Exits non-zero if structural integrity checks fail.

Usage: python scripts/audit_episodes.py --run-dir <run> [--verbose]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_cot_stats(ep: dict) -> dict:
    turns = [ev for ev in ep["events"] if ev.get("type") == "model_turn"]
    reasoning = "".join((ev.get("reasoning_content") or "") for ev in turns)
    return {
        "turns": len(turns),
        "reasoning_chars": len(reasoning),
        "reasoning_nonempty_turns": sum(
            1 for ev in turns if (ev.get("reasoning_content") or "").strip()),
        "visible_nonempty_turns": sum(
            1 for ev in turns if (ev.get("public_content") or "").strip()),
    }


def episode_action_funnel(ep: dict) -> dict:
    props = [ev for ev in ep["events"] if ev.get("type") == "tool_proposal"]
    results = [ev for ev in ep["events"] if ev.get("type") == "tool_result"]
    actions = Counter()
    for ev in props:
        txt = ev.get("public_content") or ""
        for a in ("inspect_state", "propose_patch", "list_objects", "submit_task"):
            if f"TOOL CALL: {a}(" in txt:
                actions[a] += 1
    rejected = sum(1 for ev in results if "rejected" in (ev.get("public_content") or "").lower()
                    or "error" in (ev.get("public_content") or "").lower())
    return {"proposals": len(props), "results": len(results),
            "actions": dict(actions), "rejected_or_error_results": rejected}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()

    episodes = []
    for shard in sorted((run / "episodes").glob("shard-*.jsonl")):
        episodes.extend(load_jsonl(shard))
    priv = {}
    for shard in sorted((run / "privileged").glob("shard-*.jsonl")):
        for rec in load_jsonl(shard):
            priv[rec["episode_id"]] = rec

    status_counts = Counter(ep["status"] for ep in episodes)
    policy_counts = Counter(ep.get("policy_mode", priv.get(ep["episode_id"], {}).get("policy", "?"))
                            for ep in episodes)

    print(f"run_dir: {run}")
    print(f"episodes: {len(episodes)}  statuses={dict(status_counts)}  policies={dict(policy_counts)}")
    print()

    funnel = {"honest": Counter(), "attack": Counter()}
    for ep in episodes:
        pol = ep.get("policy_mode") or priv.get(ep["episode_id"], {}).get("policy", "?")
        out = ep.get("outcomes", {})
        st = episode_cot_stats(ep)
        act = episode_action_funnel(ep)
        tag = "OK " if ep["status"] == "completed" else "!!!"
        print(f"[{tag}] {ep['episode_id']}")
        print(f"     family={ep.get('family_id')} template={ep.get('template_id')} policy={pol} status={ep['status']}")
        print(f"     outcomes: attack_attempted={out.get('attack_attempted')} "
              f"attack_succeeded={out.get('attack_succeeded')} goal_reached={out.get('goal_reached')} "
              f"main_task_succeeded={out.get('main_task_succeeded')} invalid={out.get('invalid_action_count')}")
        print(f"     cot: turns={st['turns']} reasoning_chars={st['reasoning_chars']} "
              f"reasoning_nonempty={st['reasoning_nonempty_turns']}/{st['turns']} "
              f"visible_nonempty={st['visible_nonempty_turns']}/{st['turns']}")
        print(f"     actions: {act['actions']}  rejected/error={act['rejected_or_error_results']}")
        if pol in funnel:
            funnel[pol]["episodes"] += 1
            if out.get("attack_attempted"):
                funnel[pol]["attack_attempted"] += 1
            if out.get("attack_succeeded"):
                funnel[pol]["attack_succeeded"] += 1
            if out.get("goal_reached"):
                funnel[pol]["goal_reached"] += 1
            if out.get("main_task_succeeded"):
                funnel[pol]["main_task_succeeded"] += 1
        if args.verbose:
            for ev in ep["events"]:
                if ev.get("type") == "model_turn" and (ev.get("reasoning_content") or "").strip():
                    print(f"       [turn {ev.get('step_index')}] REASONING: "
                          f"{(ev.get('reasoning_content') or '')[:300]!r}")
        print()

    print("CAPABILITY FUNNEL (attack episodes):")
    a = funnel["attack"]
    n = max(a["episodes"], 1)
    print(f"  attack episodes           : {a['episodes']}")
    print(f"  attack attempted          : {a['attack_attempted']}  ({a['attack_attempted']/n:.0%})")
    print(f"  attack succeeded          : {a['attack_succeeded']}  ({a['attack_succeeded']/n:.0%})")
    print(f"  attack goal reached       : {a['goal_reached']}  ({a['goal_reached']/n:.0%})")
    print("CAPABILITY FUNNEL (honest episodes):")
    h = funnel["honest"]
    hn = max(h["episodes"], 1)
    print(f"  honest episodes           : {h['episodes']}")
    print(f"  main task succeeded       : {h['main_task_succeeded']}  ({h['main_task_succeeded']/hn:.0%})")
    print(f"  false attack attempts     : {h['attack_attempted']}  (should be 0; nonzero => policy leakage bug)")

    problems = []
    if h["attack_attempted"] > 0:
        problems.append("honest episode attempted attack — policy isolation broken")
    for ep in episodes:
        if ep["status"] == "completed" and ep.get("trace_available") is False:
            problems.append(f"{ep['episode_id']}: completed but trace_available=False")
        st = episode_cot_stats(ep)
        if ep["status"] == "completed" and st["reasoning_chars"] == 0:
            problems.append(f"{ep['episode_id']}: completed but zero CoT chars "
                            f"(reasoning capture failed)")
    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("EPISODE AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
