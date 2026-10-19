# TRACEGUARD generation kernel.
# Pipeline: quick preflight -> clone pinned repo -> install -> load Qwen3-4B ->
#   1) one-model-turn think-token/tool-call validation -> 2) episode generation.
# Exits nonzero on any gate failure. Writes environment.json + run dir under
# /kaggle/working/traceguard-output/.

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
TRACEGUARD_COMMIT = "4b9caa0d1f236a0fb2f4a7f6196c62afd6aeec38"
REPO_URL = "https://github.com/heicfast/traceguard.git"
STAGE = "pilot"        # gpu_smoke | pilot | confirmatory | online
CONFIG = "configs/pilot.yaml"
SHARD = 1
SHARD_COUNT = 2
TORCH_DTYPE = "float16"    # T4 (sm_75): no native bf16 -> force fp16

OUT = Path("/kaggle/working/traceguard-output")
REPO_DIR = Path("/tmp/traceguard")  # cloned repo stays OUT of the output upload


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gate(condition: bool, message: str, manifest: dict) -> None:
    manifest.setdefault("gates", []).append(
        {"gate": message, "passed": bool(condition), "time_utc": datetime.now(UTC).isoformat()})
    if not condition:
        (OUT / f"{STAGE}").mkdir(parents=True, exist_ok=True)
        (OUT / f"{STAGE}" / "kernel_failure.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("GATE FAILED:", message, flush=True)
        raise SystemExit(1)
    print("GATE PASSED:", message, flush=True)


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kernel": "traceguard-gen-pilot",
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

    # ---- install: never reinstall torch; force transformers>=4.51 (qwen3) ------
    inst = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
                f"{REPO_DIR}[analysis]"], check=False)
    manifest["pip_editable_rc"] = inst.returncode
    gate(inst.returncode == 0,
         f"pip install -e rc=0 (stderr: {inst.stderr[-200:]})", manifest)
    import importlib

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

    # ---- model load + authentic think-token validation -------------------------
    os.environ["TRACEGUARD_TORCH_DTYPE"] = TORCH_DTYPE
    os.environ["TRACEGUARD_DEVICE_MAP"] = '{"": 0}'
    sys.path.insert(0, str(REPO_DIR / "src"))
    t0 = time.time()
    from traceguard.models.qwen import QwenHFProvider
    provider = QwenHFProvider(model_id="Qwen/Qwen3-4B")
    manifest["model_load_seconds"] = round(time.time() - t0, 1)
    manifest["model_revision"] = provider.model_revision
    gate(len(provider.model_revision) >= 6 and "unresolved" not in provider.model_revision,
         f"model revision resolved: {provider.model_revision}", manifest)
    free_bytes, _ = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_after_load_gib"] = round(free_bytes / 2**30, 2)
    peak = torch.cuda.max_memory_allocated() / 2**30
    manifest["gpu_peak_vram_after_load_gib"] = round(peak, 2)

    from traceguard.models.interface import GenerationSettings
    settings = GenerationSettings(temperature=0.6, top_p=0.95, top_k=20,
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

    # token/sec measurement for budget planning
    secs = turn.elapsed_ms / 1000.0
    toks = turn.token_usage.get("generated_tokens", 0)
    manifest["measured_tokens_per_sec"] = round(toks / secs, 1) if secs > 0 else None

    # capture resolved revision BEFORE releasing the validation model
    resolved_model_revision = provider.model_revision

    # ---- RELEASE the validation model before the generation subprocess loads
    # its own copy (two resident 8GB copies would OOM a single T4). -----------
    import gc
    del provider, turn
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_bytes, _total3 = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_after_release_gib"] = round(free_bytes / 2**30, 2)
    gate(free_bytes / 2**30 > 12.0, f"VRAM released for generation ({manifest['gpu_free_vram_after_release_gib']} GiB free)",
         manifest)

    # ---- write environment snapshot ---------------------------------------------
    (OUT / STAGE).mkdir(parents=True, exist_ok=True)
    (OUT / STAGE / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- config placeholder substitution ----------------------------------------
    # The frozen configs keep PIN_EXACT_COMMIT_BEFORE_RUN placeholders for
    # model/code revisions; the kernel resolves them at run time so the run
    # manifest records the TRUE as-run values (and the config hash covers the
    # actual frozen inputs, not the placeholders).
    import yaml
    cfg_data = yaml.safe_load((REPO_DIR / CONFIG).read_text(encoding="utf-8"))
    cfg_data["provenance"]["code_revision"] = TRACEGUARD_COMMIT
    cfg_data["agent"]["model_revision"] = resolved_model_revision
    cfg_data["agent"]["tokenizer_revision"] = resolved_model_revision
    cfg_data["monitor"]["model_revision"] = resolved_model_revision
    patched_cfg = Path("/tmp/run_config.yaml")
    patched_cfg.write_text(yaml.safe_dump(cfg_data, sort_keys=True), encoding="utf-8")
    manifest["config_source"] = CONFIG
    manifest["config_substituted"] = patched_cfg.name
    manifest["config_code_revision"] = TRACEGUARD_COMMIT
    manifest["config_model_revision"] = resolved_model_revision

    # ---- episode generation via CLI (frozen config, resumable shards) -----------
    gen_cmd = [sys.executable, "-m", "traceguard.cli", "generate",
               "--config", str(patched_cfg), "--run-dir", str(OUT / STAGE / "run"),
               "--stage", STAGE, "--shard", str(SHARD), "--shard-count", str(SHARD_COUNT),
               "--provider", "qwen"]
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
    print("GEN KERNEL COMPLETE", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
