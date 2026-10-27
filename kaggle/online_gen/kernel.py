# TRACEGUARD RQ4 online pre-action intervention study kernel.
# Pipeline: quick preflight -> clone pinned repo -> install -> validate 4B
#   (authentic think tokens) -> RELEASE -> substitute config pins ->
#   dataset-identity gate vs the pilot cohort -> online workers (one shared
#   4B checkpoint per GPU serving agent + fusion monitor) -> merge + verify.
# Exits nonzero on any gate failure. Writes environment.json + the online run
# under /kaggle/working/traceguard-output/.

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
TRACEGUARD_COMMIT = "PIN_BEFORE_PUSH"
REPO_URL = "https://github.com/heicfast/traceguard.git"
CONFIG = "configs/online_study.yaml"
PLAN_SOURCE_CONFIG = "configs/phase3_pilot.yaml"
SHARD = 0
SHARD_COUNT = 1
TORCH_DTYPE = "float16"    # T4 (sm_75): no native bf16 -> force fp16

OUT = Path("/kaggle/working/traceguard-output")
REPO_DIR = Path("/tmp/traceguard")  # cloned repo stays OUT of the output upload


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gate(condition: bool, message: str, manifest: dict) -> None:
    manifest.setdefault("gates", []).append(
        {"gate": message, "passed": bool(condition),
         "time_utc": datetime.now(UTC).isoformat()})
    if not condition:
        (OUT / "online").mkdir(parents=True, exist_ok=True)
        (OUT / "online" / "kernel_failure.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("GATE FAILED:", message, flush=True)
        raise SystemExit(1)
    print("GATE PASSED:", message, flush=True)


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kernel": "traceguard-online-study",
        "stage": "online",
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
    manifest["gpu"] = {"name": prop.name,
                       "capability": list(torch.cuda.get_device_capability(0)),
                       "vram_gib": round(prop.total_memory / 2**30, 3),
                       "count": torch.cuda.device_count()}
    free_bytes, _total2 = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_at_start_gib"] = round(free_bytes / 2**30, 2)
    print(json.dumps(manifest["gpu"]), flush=True)

    # ---- repo clone at pinned commit ------------------------------------------
    r = run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)])
    gate(r.returncode == 0, f"git clone ({r.stderr[:200]})", manifest)
    r = run(["git", "-C", str(REPO_DIR), "checkout", TRACEGUARD_COMMIT])
    gate(r.returncode == 0, f"checkout {TRACEGUARD_COMMIT[:10]}", manifest)

    # ---- install (never reinstall torch; transformers>=4.51 for qwen3) --------
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
        gate(up.returncode == 0, f"upgrade transformers ({up.stderr[:200:]})", manifest)
        importlib.invalidate_caches()
    import transformers as tfs
    manifest["transformers_used"] = tfs.__version__
    gate(vparse(tfs.__version__) >= vparse("4.51"),
         f"transformers {tfs.__version__} >= 4.51", manifest)

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
    secs = turn.elapsed_ms / 1000.0
    toks = turn.token_usage.get("generated_tokens", 0)
    manifest["measured_tokens_per_sec"] = round(toks / secs, 1) if secs > 0 else None
    resolved_model_revision = provider.model_revision

    # ---- RELEASE the validation model before the online workers load their own
    import gc
    del provider, turn
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_bytes, _ = torch.cuda.mem_get_info()
    manifest["gpu_free_vram_after_release_gib"] = round(free_bytes / 2**30, 2)
    gate(free_bytes / 2**30 > 12.0,
         f"VRAM released for online workers ({manifest['gpu_free_vram_after_release_gib']} GiB free)",
         manifest)

    # ---- environment snapshot ---------------------------------------------------
    (OUT / "online").mkdir(parents=True, exist_ok=True)
    (OUT / "online" / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- config pin substitution + dataset-identity gate -----------------------
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
    manifest["frozen_threshold"] = cfg_data["monitor"]["online_blocking_threshold"]

    # The paired design REQUIRES the online arm to replay the pilot cohort's
    # frozen plan: the dataset sections of the online and pilot configs must
    # be identical, and the agent sections must match too.
    pilot_cfg = yaml.safe_load((REPO_DIR / PLAN_SOURCE_CONFIG).read_text(encoding="utf-8"))
    gate(cfg_data["dataset"] == pilot_cfg["dataset"],
         "dataset section identical to the pilot cohort config", manifest)
    gate(cfg_data["agent"] == pilot_cfg["agent"],
         "agent section identical to the pilot cohort config", manifest)
    gate(cfg_data["monitor"]["online_blocking_threshold"] is not None,
         "frozen blocking threshold present in config", manifest)

    # ---- online study: one worker per visible GPU -------------------------------
    online_cmd = [sys.executable, str(REPO_DIR / "scripts" / "run_online_workers.py"),
                  "--config", str(patched_cfg), "--output", str(OUT / "online"),
                  "--kernel-shard", str(SHARD),
                  "--kernel-shard-count", str(SHARD_COUNT)]
    proc = subprocess.run(online_cmd, cwd=str(REPO_DIR))
    manifest["online_rc"] = proc.returncode
    (OUT / "online" / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate(proc.returncode == 0, "online study completed (workers + merge + verify)", manifest)

    # ---- copy the worker logs next to the run for provenance --------------------
    workers_src = OUT / "online" / "workers"
    if workers_src.exists():
        logs_dst = OUT / "online" / "online-worker-logs"
        logs_dst.mkdir(parents=True, exist_ok=True)
        for log in workers_src.rglob("online.log"):
            rel = log.relative_to(workers_src)
            shutil.copy2(log, logs_dst / str(rel).replace("/", "_"))

    peak = torch.cuda.max_memory_allocated() / 2**30
    manifest["gpu_peak_vram_final_gib"] = round(peak, 2)
    manifest["kernel_seconds"] = round(time.time() - started, 1)
    (OUT / "online" / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ONLINE KERNEL COMPLETE", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
