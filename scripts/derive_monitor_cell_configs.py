#!/usr/bin/env python3
"""Derive frozen monitor-cell configs from the phase-3 pilot's run manifest.

The cell configs keep every generation-defining section byte-identical to the
frozen generation config (seed/agent/sandbox/dataset + the checked provenance
fields) and swap ONLY the monitor model selection + matrix_cell_id. This is the
monitor-capability / monitor-family axis of the phase-3 matrix: the SAME stored
episodes are re-scored by a different monitor, so any detection difference is
attributable to the monitor, not the episodes.

Usage: python scripts/derive_monitor_cell_configs.py
Writes: configs/phase3_monitor_cells/{scale_monitor_qwen14,cross_monitor_deepseek14}.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MANIFEST = REPO / "runs/v6/pilot_v0_run/run_manifest.json"
OUT_DIR = REPO / "configs/phase3_monitor_cells"

# Admitted monitor candidates: revisions + probe hashes are read from the
# COMMITTED v0.6 admission artifacts (runs/v6/probes/monitor/) — never
# hardcoded, so a fresh admission automatically propagates.
def _admission(filename: str) -> tuple[str, str]:
    artifact = json.loads((REPO / "runs/v6/probes/monitor" / filename).read_text(encoding="utf-8"))
    assert artifact.get("all_gates_passed") is True, filename
    assert artifact.get("role") == "monitor", filename
    return artifact["model_revision"], artifact["artifact_sha256"]

QWEN3_14B_REV, QWEN3_14B_PROBE = _admission("qwen3_14b_monitor_probe_artifact.json")
DEEPSEEK_R1_14B_REV, DEEPSEEK_R1_14B_PROBE = _admission("deepseek_r1_14b_monitor_probe_artifact.json")

CELLS = {
    "scale_monitor_qwen14": {
        "cell_id": "scale-monitor-qwen4-qwen14",
        "monitor_swap": {
            "model_id": "Qwen/Qwen3-14B",
            "model_revision": QWEN3_14B_REV,
            "model_family": "qwen3",
            "provider": "qwen_hf",
            "reasoning_protocol": "qwen3",
            "quantization": "bnb_nf4",
            "compute_dtype": "float16",
            "attention_backend": "sdpa",
            "probe_artifact_sha256": QWEN3_14B_PROBE,
        },
    },
    "cross_monitor_deepseek14": {
        "cell_id": "cross-monitor-qwen4-deepseek14",
        "monitor_swap": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "model_revision": DEEPSEEK_R1_14B_REV,
            "model_family": "qwen2_5_r1_distill",
            "provider": "generic_hf",
            "reasoning_protocol": "think_tags",
            "quantization": "bnb_nf4",
            "compute_dtype": "float16",
            "attention_backend": "sdpa",
            "probe_artifact_sha256": DEEPSEEK_R1_14B_PROBE,
        },
    },
}


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base_config = manifest["config"]
    if str(base_config["provenance"]["code_revision"]).startswith("PIN"):
        raise SystemExit("generation manifest carries an unresolved PIN; refusing to derive cells")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from traceguard.config import load_config

    for name, cell in CELLS.items():
        payload = yaml.safe_load(yaml.safe_dump(base_config))  # deep copy via round-trip
        payload["monitor"].update(cell["monitor_swap"])
        # matrix_cell_id is the ONLY provenance field added: it is excluded from
        # the generation-consistency check and drives the suffixed monitor
        # identity (llm_<provider>_<hash10>) so cell scores can never collide
        # with anchor-cell scores.
        payload["provenance"]["matrix_cell_id"] = cell["cell_id"]

        path = OUT_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

        # Validate: schema + the exact runtime consistency contract the kernel
        # will enforce (generation-defining sections + checked provenance).
        cfg = load_config(str(path))
        gen = manifest["config"]
        for section in ("seed", "agent", "sandbox", "dataset"):
            if gen[section] != cfg.to_payload()[section]:
                raise SystemExit(f"{name}: section {section} drifted from the frozen generation config")
        for key in ("code_revision", "experiment_name", "experiment_stage", "replicate_id"):
            if gen["provenance"][key] != cfg.to_payload()["provenance"][key]:
                raise SystemExit(f"{name}: provenance {key} drifted")
        if cfg.to_payload()["provenance"]["matrix_cell_id"] != cell["cell_id"]:
            raise SystemExit(f"{name}: matrix_cell_id not set")

        from traceguard.ids import sha256_text
        identity = "|".join([
            cfg.monitor.model_id, cfg.monitor.model_revision,
            cfg.monitor.adapter_id or "", cfg.monitor.adapter_revision or "",
        ])
        monitor_name = f"llm_{cfg.monitor.provider}_{sha256_text(identity)[:10]}"
        print(f"OK {path.name}: cell={cell['cell_id']} monitor_name={monitor_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
