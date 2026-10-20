#!/usr/bin/env python3
"""Derive frozen agent-cell GENERATION configs from the phase-3 anchor config.

The agent-capability axis of the matrix needs NEW cohorts (a stronger agent
generates its own episodes): scale-agent (Qwen3-14B) and cross-agent
(DeepSeek-R1-Distill-Qwen-14B), both monitored by the SAME frozen 4B anchor
monitor. Unlike the monitor cells, these configs cannot be derived from the
anchor run manifest alone — the agent section swaps to a probe-admitted 14B
checkpoint, so this script REFUSES to run until the agent-role probe
artifacts exist and pass.

Cohort sizing (declared, deviation-recorded): 64 episodes per agent cohort
(seeds_per_template_per_policy: 1) so a dual-T4 kernel generates a cohort in
one session safely under the 12-hour Kaggle cap (14B NF4 is ~3x slower than
the 4B anchor; the anchor's 128-episode cohort took 2.6 h).

Everything else is copied verbatim from the anchor run's frozen config; the
monitor keeps the anchor's RESOLVED 4B revision so agent cells score under
the identical monitor checkpoint.

Usage: python scripts/derive_agent_cell_configs.py <freeze_commit>
Writes: configs/phase3_agent_cells/{scale_agent_qwen14,cross_agent_deepseek14}.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ANCHOR_MANIFEST = REPO / "runs/v6/pilot_v0_run/run_manifest.json"
PROBE_DIR = REPO / "runs/v6/probes"
OUT_DIR = REPO / "configs/phase3_agent_cells"

# Same checkpoints admitted for the monitor role; the AGENT-role admission is
# what this script consumes (probe kernels: traceguard-probe-qwen14-agent /
# traceguard-probe-deepseek14-agent).
CELLS = {
    "scale_agent_qwen14": {
        "cell_id": "generation:agent-qwen14",
        "experiment_name": "traceguard_phase3_agent_qwen14_v1",
        "probe_artifact": "agent/qwen3_14b_agent_probe_artifact.json",
        "agent_swap": {
            "model_id": "Qwen/Qwen3-14B",
            "provider": "qwen_hf",
            "reasoning_protocol": "qwen3",
            "quantization": "bnb_nf4",
        },
    },
    "cross_agent_deepseek14": {
        "cell_id": "generation:agent-deepseek14",
        "experiment_name": "traceguard_phase3_agent_deepseek14_v1",
        "probe_artifact": "agent/deepseek_r1_14b_agent_probe_artifact.json",
        "agent_swap": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "provider": "generic_hf",
            "reasoning_protocol": "think_tags",
            "quantization": "bnb_nf4",
        },
    },
}


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: derive_agent_cell_configs.py <freeze_commit_sha> [cell_name ...]")
    freeze_commit = sys.argv[1]
    only = sys.argv[2:] or None
    if len(freeze_commit) != 40 or any(c not in "0123456789abcdef" for c in freeze_commit):
        raise SystemExit(f"freeze commit must be a full 40-char sha, got {freeze_commit!r}")

    manifest = json.loads(ANCHOR_MANIFEST.read_text(encoding="utf-8"))
    base = manifest["config"]
    resolved_monitor_rev = base["monitor"]["model_revision"]
    if str(resolved_monitor_rev).startswith("PIN"):
        raise SystemExit("anchor manifest monitor revision unresolved; refusing to derive")

    from traceguard.config import load_config

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cell in CELLS.items():
        if only and name not in only:
            continue
        artifact_path = PROBE_DIR / cell["probe_artifact"]
        if not artifact_path.exists():
            raise SystemExit(
                f"{name}: agent probe artifact missing ({artifact_path}) — run the "
                f"agent-role admission probe first; no config is derived without it")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not artifact.get("all_gates_passed"):
            raise SystemExit(f"{name}: agent probe did not pass all gates; refusing admission")
        revision = artifact["model_revision"]
        if not revision or str(revision).startswith("PIN"):
            raise SystemExit(f"{name}: probe artifact revision unresolved")

        payload = yaml.safe_load(yaml.safe_dump(base))
        payload["agent"].update(cell["agent_swap"])
        payload["agent"]["model_revision"] = revision
        payload["agent"]["tokenizer_revision"] = revision
        payload["agent"]["probe_artifact_sha256"] = artifact["artifact_sha256"]
        # 64-episode cohorts (T4 session-cap feasibility; deviation-recorded).
        payload["dataset"]["seeds_per_template_per_policy"] = 1
        payload["provenance"].update({
            "code_revision": freeze_commit,
            "experiment_name": cell["experiment_name"],
            "matrix_cell_id": cell["cell_id"],
        })
        # monitor stays the anchor's resolved 4B checkpoint — byte-identical
        # section from the frozen manifest (no substitution at kernel time).

        path = OUT_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

        cfg = load_config(str(path))
        if cfg.agent.model_id != cell["agent_swap"]["model_id"]:
            raise SystemExit(f"{name}: agent swap failed")
        if cfg.agent.probe_artifact_sha256 != artifact["artifact_sha256"]:
            raise SystemExit(f"{name}: probe hash not bound")
        if cfg.monitor.model_revision != resolved_monitor_rev:
            raise SystemExit(f"{name}: monitor revision drifted from the anchor checkpoint")
        if cfg.dataset.seeds_per_template_per_policy != 1:
            raise SystemExit(f"{name}: cohort sizing not applied")
        # catalog admission for the agent role (probe-required + admitted path)
        from traceguard.models.catalog import validate_model_assignment
        validate_model_assignment(cfg.agent.model_id, role="agent",
                                  quantization=cfg.agent.quantization,
                                  allow_probe_required=True)
        print(f"OK {path.name}: agent={cfg.agent.model_id}@{revision[:10]} "
              f"monitor={cfg.monitor.model_id}@{resolved_monitor_rev[:10]} "
              f"probe={artifact['artifact_sha256'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
