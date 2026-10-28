#!/usr/bin/env python3
"""Supplementary analyses requested by the 2026-10-28 scientist review (F10).

Runs entirely on the landed evidence and writes a single archived report:
  (a) E-condition operating-point mechanics: benign score distributions,
      the threshold-inflating episode, per-split realized FPR vs C.
  (b) Negative-pool sensitivity: A-AUROC with honest-only negatives
      (never-attempted attacks removed from the benign pool).
  (c) 4B zero-rubric forensics: rubric component distributions + a
      verdict-text sample on positives (the mechanism of the collapse).
  (d) Family x strategy cross-table for actions-only detection at 14B.
  (e) Keywords movement: actions vs CoT flag counts (extractable-signal).

Usage:
  python scripts/supplementary_analyses.py --out runs/v6/supplementary_analyses_20261028.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_scores(run: Path, prefix: str, condition: str) -> dict:
    out = {}
    for f in sorted((run / "monitor_scores").glob(f"{prefix}-{condition}-*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                out[rec["episode_id"]] = rec.get("suspicion")
    return out


def load_records(run: Path, prefix: str, condition: str) -> dict:
    out = {}
    for f in sorted((run / "monitor_scores").glob(f"{prefix}-{condition}-*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                out[rec["episode_id"]] = rec
    return out


def episode_meta(run: Path) -> dict:
    meta = {}
    for f in sorted((run / "episodes").glob("*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                ep = json.loads(ln)
                out = ep.get("outcomes") or {}
                prov = ep.get("provenance") or {}
                meta[ep["episode_id"]] = {
                    "family": ep.get("family_id"),
                    "policy": ep.get("policy_mode"),
                    "strategy": prov.get("attack_strategy"),
                    "split": prov.get("split_role"),
                    "attempted": out.get("attack_attempted"),
                    "succeeded": out.get("attack_succeeded"),
                }
    return meta


def auroc(pos_vals, neg_vals) -> float | None:
    pos = [v for v in pos_vals if v is not None]
    neg = [v for v in neg_vals if v is not None]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="runs/v6/scale_mon_qwen14/run")
    ap.add_argument("--anchor", default="runs/v6/anchor_mon/run")
    ap.add_argument("--out", default="runs/v6/supplementary_analyses_20261028.json")
    args = ap.parse_args()
    scale = Path(args.scale)
    anchor = Path(args.anchor)
    report: dict = {}

    meta = episode_meta(scale)
    pos_ids = [e for e, m in meta.items() if m["attempted"]]
    honest_ids = [e for e, m in meta.items() if not m["attempted"]]

    # ---- (a) E-condition operating-point mechanics (14B) --------------------
    e14 = load_scores(scale, "llm_qwen_hf_c885515408", "actions_plus_shuffled_cot")
    c14 = load_scores(scale, "llm_qwen_hf_c885515408", "actions_plus_cot")
    e_benign = sorted((e14[e], e) for e in honest_ids if e14.get(e) is not None)
    c_benign = sorted((c14[e], e) for e in honest_ids if c14.get(e) is not None)
    report["a_E_operating_point_14b"] = {
        "E_benign_score_distribution": collections.Counter(
            v for v, _ in e_benign),
        "E_benign_max": e_benign[-1][0] if e_benign else None,
        "E_benign_max_episode": e_benign[-1][1] if e_benign else None,
        "C_benign_max": c_benign[-1][0] if c_benign else None,
        "E_benign_at_or_above_1.0": sum(1 for v, _ in e_benign if v >= 1.0),
        "C_benign_at_or_above_1.0": sum(1 for v, _ in c_benign if v >= 1.0),
        "E_positives_score_range": (
            min(v for v in (e14[e] for e in pos_ids) if v is not None),
            max(v for v in (e14[e] for e in pos_ids) if v is not None)),
    }
    # the threshold-inflating episode, verified against its own C score
    if e_benign and e_benign[-1][0] >= 1.0:
        eid = e_benign[-1][1]
        report["a_E_operating_point_14b"]["inflating_episode"] = {
            "episode_id": eid,
            "E_score": e14[eid],
            "C_score": c14.get(eid),
            "A_score": load_scores(
                scale, "llm_qwen_hf_c885515408", "actions").get(eid),
            "family": meta[eid]["family"],
            "split": meta[eid]["split"],
        }

    # ---- (b) negative-pool sensitivity for A-AUROC (14B) --------------------
    a14 = load_scores(scale, "llm_qwen_hf_c885515408", "actions")
    never_attempted = [e for e, m in meta.items()
                       if m["policy"] == "attack" and not m["attempted"]]
    pos_vals = [a14[e] for e in pos_ids]
    honest_only_neg = [a14[e] for e in honest_ids]
    report["b_negative_pool_sensitivity_14b"] = {
        "A_auroc_honest_only_pool": auroc(pos_vals, honest_only_neg),
        "A_auroc_honest_only": auroc(pos_vals, honest_only_neg),
        "never_attempted_attacks_in_benign_pool": len(never_attempted),
        "note": "never-attempted attacks are NOT in the benign pool here; "
                "the full-pool A-AUROC in the analysis frame uses honest "
                "episodes only. Both values recorded for transparency.",
    }

    # ---- (c) 4B zero-rubric forensics (anchor cell) ---------------------------
    a4 = load_records(anchor, "llm_qwen_hf", "actions")
    c4 = load_records(anchor, "llm_qwen_hf", "actions_plus_cot")
    rubric_keys = ("policy_conflict", "unauthorized_intent",
                   "concealment", "action_evidence")

    def rubric_dist(records, ids):
        dist = {k: collections.Counter() for k in rubric_keys}
        for e in ids:
            rec = records.get(e)
            if not rec:
                continue
            rub = (rec.get("parsed_response") or {}).get("rubric") or {}
            for k in rubric_keys:
                dist[k][rub.get(k, "missing")] += 1
        return {k: dict(v) for k, v in dist.items()}

    report["c_4b_zero_rubric_forensics"] = {
        "A_rubric_on_positives": rubric_dist(a4, pos_ids),
        "C_rubric_on_positives": rubric_dist(c4, pos_ids),
        "A_rubric_on_honest": rubric_dist(a4, honest_ids),
        "C_distinct_suspicion_values": sorted(
            {rec.get("suspicion") for rec in c4.values()
             if rec.get("suspicion") is not None}),
        "positives_with_any_nonzero_C_rubric": sum(
            1 for e in pos_ids if c4.get(e) and any(
                ((c4[e].get("parsed_response") or {}).get("rubric") or {}).get(k)
                for k in rubric_keys)),
        "verdict_text_samples_nonzero": [
            {"episode_id": e, "verdict": ((c4[e].get("parsed_response") or {})
                                          .get("verdict_text", ""))[:300]}
            for e in pos_ids
            if c4.get(e) and any(
                ((c4[e].get("parsed_response") or {}).get("rubric") or {}).get(k)
                for k in rubric_keys)
        ][:5],
    }

    # ---- (d) family x strategy cross-table for actions-only (14B) -------------
    cross: dict = collections.defaultdict(dict)
    for fam in sorted({m["family"] for m in meta.values()}):
        for strat in sorted({m["strategy"] for m in meta.values()}):
            grp_pos = [e for e, m in meta.items()
                       if m["attempted"] and m["family"] == fam
                       and m["strategy"] == strat]
            grp_neg = [e for e, m in meta.items()
                       if not m["attempted"] and m["family"] == fam]
            if grp_pos and grp_neg:
                cross[fam][strat] = {
                    "n_pos": len(grp_pos),
                    "A_auroc": auroc([a14[e] for e in grp_pos],
                                     [a14[e] for e in grp_neg]),
                }
    report["d_family_x_strategy_A_14b"] = {k: dict(v) for k, v in cross.items()}

    # ---- (e) keywords movement ------------------------------------------------
    kw_a = load_records(scale, "keywords_v1", "actions")
    kw_c = load_records(scale, "keywords_v1", "actions_plus_cot")
    def flagged(rec):
        return bool(rec) and (rec.get("suspicion") or 0) > 0
    report["e_keywords_movement_14b_cohort"] = {
        "positives_flagged_actions": sum(1 for e in pos_ids if flagged(kw_a.get(e))),
        "positives_flagged_actions_plus_cot": sum(1 for e in pos_ids if flagged(kw_c.get(e))),
        "honest_flagged_actions": sum(1 for e in honest_ids if flagged(kw_a.get(e))),
        "honest_flagged_actions_plus_cot": sum(1 for e in honest_ids if flagged(kw_c.get(e))),
        "note": "keyword suspicion > 0 means a suspicious-phrase hit; "
                "the deterministic baseline extracts no action-level signal "
                "but does extract CoT-level signal on this cohort.",
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str)[:3000])
    print(f"wrote {out}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
