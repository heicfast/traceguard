# TRACEGUARD agent-cell generation kernel: cross-agent-deepseek14-qwen4.
# Generates the 64-episode agent-capability cohort with the probe-admitted
# CROSS-FAMILY agent (DeepSeek-R1-Distill-Qwen-14B, NF4, think_tags) under the frozen phase-3 task bank; the
# cohort is later scored by the SAME frozen 4B anchor monitor (weak-to-strong
# stress axis). Config: configs/phase3_agent_cells/cross_agent_deepseek14.yaml —
# fully resolved (no PIN placeholders), derived by
# scripts/derive_agent_cell_configs.py from the anchor run manifest with the
# agent-role probe artifact bound.

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ---------------- run parameters (edited per push; recorded in output) ---------
TRACEGUARD_COMMIT = "FREEZE_COMMIT_PLACEHOLDER"
REPO_URL = "https://github.com/heicfast/traceguard.git"
STAGE = "pilot"
CONFIG = "configs/phase3_agent_cells/cross_agent_deepseek14.yaml"
CELL_ID = "generation:agent-deepseek14"
SHARD = 0
SHARD_COUNT = 1
TORCH_DTYPE = "float16"    # T4 (sm_75): no native bf16 -> force fp16

OUT = Path("/kaggle/working/traceguard-output")
REPO_DIR = Path("/tmp/traceguard")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gate(condition: bool, message: str, manifest: dict) -> None:
    manifest.setdefault("gates", []).append(
        {"gate": message, "passed": bool(condition), "time_utc": datetime.now(UTC).isoformat()})
    if not condition:
        (OUT / STAGE).mkdir(parents=True, exist_ok=True)
        (OUT / STAGE / "kernel_failure.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("GATE FAILED:", message, flush=True)
        raise SystemExit(1)
    print("GATE PASSED:", message, flush=True)


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kernel": "traceguard-gen-cell",
        "cell_id": CELL_ID,
        "stage": STAGE,
        "config": CONFIG,
        "shard": f"{SHARD}/{SHARD_COUNT}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "pinned_commit": TRACEGUARD_COMMIT,
        "gates": [],
    }
    total, used, free = shutil.disk_usage("/kaggle/working")
    manifest["disk_free_gib"] = round(free / 2**30, 1)

    # ---- gate 1: torch + GPU present with working fp16 kernels ---------------
    import torch
    manifest["torch"] = torch.__version__
    manifest["cuda_build"] = torch.version.cuda
    gate(torch.cuda.is_available(), "cuda available", manifest)
    device = torch.device("cuda:0")
    x = torch.randn((512, 512), device=device, dtype=torch.float16)
    val = (x @ x).float().sum().item()
    gate(val == val and abs(val) != float("inf"), "fp16 matmul/reduction finite", manifest)
    prop = torch.cuda.get_device_properties(0)
    manifest["gpu"] = {"name": prop.name, "capability": list(torch.cuda.get_device_capability(0)),
                       "vram_gib": round(prop.total_memory / 2**30, 3)}
    free_bytes, _total2 = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_at_start_gib"] = round(free_bytes / 2**30, 2)
    print(json.dumps(manifest["gpu"]), flush=True)

    # ---- repo clone at pinned commit ------------------------------------------
    r = run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)])
    gate(r.returncode == 0, f"git clone ({r.stderr[:200]})", manifest)
    r = run(["git", "-C", str(REPO_DIR), "checkout", TRACEGUARD_COMMIT])
    gate(r.returncode == 0, f"checkout {TRACEGUARD_COMMIT[:10]}", manifest)

    # ---- install: repo (no deps) + bitsandbytes for NF4 ------------------------
    r = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
             f"{REPO_DIR}[analysis]"], check=False)
    manifest["pip_repo_rc"] = r.returncode
    gate(r.returncode == 0, f"pip install -e rc=0 (stderr: {r.stderr[-200:]})", manifest)
    import importlib
    import importlib.util

    import transformers
    manifest["transformers_before"] = transformers.__version__
    from packaging.version import parse as vparse
    if vparse(transformers.__version__) < vparse("4.51"):
        up = run([sys.executable, "-m", "pip", "install", "-q", "-U",
                  "transformers>=4.51,<5"], check=False)
        gate(up.returncode == 0, f"upgrade transformers ({up.stderr[-200:]})", manifest)
        importlib.invalidate_caches()
    import transformers as tfs
    manifest["transformers_used"] = tfs.__version__
    gate(vparse(tfs.__version__) >= vparse("4.51"), f"transformers {tfs.__version__} >= 4.51",
         manifest)

    if importlib.util.find_spec("bitsandbytes") is None:
        r = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
        manifest["pip_bnb_rc"] = r.returncode
        gate(r.returncode == 0, "install bitsandbytes", manifest)
    import bitsandbytes  # noqa: F401
    manifest["bitsandbytes_version"] = getattr(bitsandbytes, "__version__", "unknown")

    # ---- config: fully resolved, no substitutions ------------------------------
    import yaml
    os.environ["TRACEGUARD_TORCH_DTYPE"] = TORCH_DTYPE
    os.environ["TRACEGUARD_DEVICE_MAP"] = '{"": 0}'
    sys.path.insert(0, str(REPO_DIR / "src"))
    cfg_path = REPO_DIR / CONFIG
    gate(cfg_path.exists(), f"cell config present at pin ({CONFIG})", manifest)
    cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    resolved_ok = (
        not str(cfg_data["agent"]["model_revision"]).startswith("PIN")
        and not str(cfg_data["monitor"]["model_revision"]).startswith("PIN")
        and cfg_data["provenance"]["code_revision"] == TRACEGUARD_COMMIT
        and cfg_data["provenance"]["matrix_cell_id"] == CELL_ID
    )
    gate(resolved_ok, "cell config fully resolved + pinned to this commit", manifest)
    manifest["config_model_revision"] = cfg_data["agent"]["model_revision"]
    manifest["config_monitor_revision"] = cfg_data["monitor"]["model_revision"]
    from types import SimpleNamespace
    agent_spec = SimpleNamespace(**cfg_data["agent"])
    # Admission artifact for the probe-required agent model: the factory
    # recomputes this file's hash and compares it to the config's frozen
    # probe_artifact_sha256 before constructing the provider (fail-closed).
    artifact_rel = "runs/probes/agent/deepseek_r1_14b_agent_probe_artifact.json"
    artifact_path = REPO_DIR / artifact_rel
    gate(artifact_path.exists(),
         f"agent probe artifact present at pin ({artifact_rel})", manifest)
    os.environ["TRACEGUARD_AGENT_PROBE_ARTIFACT"] = str(artifact_path)
    patched_cfg = Path("/tmp/run_config.yaml")
    patched_cfg.write_text(yaml.safe_dump(cfg_data, sort_keys=True), encoding="utf-8")

    # ---- model load via factory (admission + probe hash + pinned revision) ----
    t0 = time.time()
    from traceguard.models.factory import build_hf_provider
    provider = build_hf_provider(agent_spec, role="agent")
    manifest["model_load_seconds"] = round(time.time() - t0, 1)
    manifest["agent_model"] = cfg_data["agent"]["model_id"]
    manifest["model_revision"] = provider.model_revision
    gate(provider.model_revision == cfg_data["agent"]["model_revision"],
         f"agent revision pinned ({provider.model_revision[:10]})", manifest)
    free_bytes, _ = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_after_load_gib"] = round(free_bytes / 2**30, 2)
    peak = torch.cuda.max_memory_allocated() / 2**30
    manifest["gpu_peak_vram_after_load_gib"] = round(peak, 2)

    from traceguard.models.interface import GenerationSettings
    settings = GenerationSettings(
        temperature=cfg_data["agent"]["temperature"],
        top_p=cfg_data["agent"]["top_p"], top_k=cfg_data["agent"]["top_k"],
        max_new_tokens=512, seed=1, reasoning_mode=True)
    turn = provider.generate(
        [{"role": "user", "content": "In one short sentence: what is 17*23?"}],
        settings)
    manifest["think_validation"] = {
        "reasoning_parse_status": turn.reasoning_parse_status,
        "reasoning_chars": len(turn.reasoning_text or ""),
        "answer_excerpt": (turn.assistant_text or "")[:200],
        "generated_tokens": turn.token_usage.get("generated_tokens"),
        "elapsed_ms": round(turn.elapsed_ms),
    }
    gate(turn.reasoning_parse_status in ("complete", "truncated"),
         f"authentic think tokens captured ({turn.reasoning_parse_status})", manifest)
    gate(len(turn.assistant_text.strip()) > 0, "non-empty visible answer", manifest)

    secs = turn.elapsed_ms / 1000.0
    toks = turn.token_usage.get("generated_tokens", 0)
    manifest["measured_tokens_per_sec"] = round(toks / secs, 1) if secs > 0 else None

    # ---- RELEASE the validation model before the generation workers load ------
    import gc
    del provider, turn
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_bytes, _total3 = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_after_release_gib"] = round(free_bytes / 2**30, 2)
    gate(free_bytes / 2**30 > 12.0,
         f"VRAM released for generation ({manifest['gpu_free_vram_after_release_gib']} GiB free)",
         manifest)

    # ---- write environment snapshot ---------------------------------------------
    (OUT / STAGE).mkdir(parents=True, exist_ok=True)
    (OUT / STAGE / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- episode generation: one isolated worker per visible GPU ----------------
    gen_cmd = [sys.executable, str(REPO_DIR / "scripts" / "run_gpu_workers.py"),
               "--config", str(patched_cfg), "--output", str(OUT / STAGE),
               "--stage", STAGE, "--kernel-shard", str(SHARD),
               "--kernel-shard-count", str(SHARD_COUNT), "--provider", "qwen"]
    proc = subprocess.run(gen_cmd, cwd=str(REPO_DIR))
    gen_ok = proc.returncode == 0
    manifest["generate_rc"] = proc.returncode
    (OUT / STAGE / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate(gen_ok, "episode generation completed", manifest)

    # final GPU state
    peak = torch.cuda.max_memory_allocated() / 2**30
    manifest["gpu_peak_vram_final_gib"] = round(peak, 2)
    manifest["kernel_seconds"] = round(time.time() - started, 1)
    (OUT / STAGE / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GEN CELL KERNEL COMPLETE", json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
