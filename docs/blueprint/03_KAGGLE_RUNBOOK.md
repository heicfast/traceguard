# 03 — Kaggle GPU Runbook

**Purpose:** execute inference reproducibly under free, interruptible GPU sessions while keeping all scientific logic in testable Python modules.

## 1. Verified service characteristics and caveats

Kaggle documents GPU environments including P100 and T4×2, GPU quotas that vary with availability, about 12 hours maximum per CPU/GPU execution session in its currently published notebook guide, and persistent output under `/kaggle/working` on a saved notebook version. Check the actual current runtime and account quota before every experiment. This is a *planning assumption*, not a guaranteed allocation. Official references: https://www.kaggle.com/docs/notebooks and https://www.kaggle.com/docs/efficient-gpu-usage .

Hardware choice: prefer **T4** for contemporary PyTorch compatibility. A March–June 2026 Kaggle community issue documented PyTorch CUDA-build incompatibilities for P100 (`sm_60`) in some default images. Always run real matrix-multiply/reduction kernel preflights, not just `torch.cuda.is_available()`. Issue: https://github.com/Kaggle/docker-python/issues/1546 .

## 2. Why Kaggle changes our architecture

No Docker daemon or long-lived server may be assumed. Kaggle interactive sessions reset, so all tasks must checkpoint, resume, and export results. Each saved run should be a batch job with frozen config; no manual notebook edits between conditions. Do not place secrets in notebooks or public outputs.

Use Kaggle for **generating agent episodes and scoring monitor evidence**. Generate task manifests and perform primary analysis locally on CPU when convenient. Storage plan: import repository snapshot or pinned Git commit, use Kaggle Dataset/Model inputs for heavy static assets, write run shards to `/kaggle/working`, and export saved outputs before restarting. Do not assume `/kaggle/working` will remain available across unsaved sessions.

## 3. GPU preflight — first notebook cell

Run with `Accelerator -> GPU`, after manually checking quota. Alternatively run the packaged, tested-for-syntax `python scripts/gpu_preflight.py` (which exports a JSON environment manifest). These commands are intentional environment probes; they do not assume a fixed GPU model:

```python
import os, platform, subprocess
import torch
print('python:', platform.python_version())
print('torch:', torch.__version__)
print('cuda build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    print('VRAM (GiB):', round(torch.cuda.get_device_properties(0).total_memory/2**30, 2))
    x = torch.randn(512, 512, device=device, dtype=torch.float16)
    y = (x @ x).float().sum()
    print('GPU math sanity:', y.item())
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout[:2000])
```

If this fails, fix environment compatibility before downloading gigabytes of weights. Do not apply random CUDA downgrades; select a PyTorch build that supports the actual GPU architecture, document the exact install, restart the kernel, and repeat. Avoid mixing two PyTorch installs or installing after importing `torch` in the same process.

## 4. Model selection and allocation

- **Primary:** `Qwen/Qwen3-4B`, Apache 2.0. Its model card explicitly supports generated thinking/non-thinking modes via `tokenizer.apply_chat_template(..., enable_thinking=True)`. The study captures native model-generated thinking tokens, not a separate self-explanation. https://huggingface.co/Qwen/Qwen3-4B .
- **Smoke/fallback:** `Qwen/Qwen3-0.6B` for parser, runtime, and invalid-action tests, with **no assumption** that its behavior will yield valid science for RQ1. If a 4B model fails to pursue tasks or attacks, treat that as a capability limitation and potentially change the organism in a new registered version.
- **Memory:** 4B parameters × 2 bytes ≈ 8 GB model weights in fp16 before KV cache, temporary tensors, tokenizer, and runtime overhead. 4-bit quantization roughly decreases *weight* memory, not total allocation proportionately. Treat VRAM feasibility as something to empirically test. A two-T4 environment does not necessarily provide one unified 32-GB allocation.
- **Batching:** start with batch size 1, small prompts, 1024–2048 total generated tokens/turn, limited agent steps; benchmark peak VRAM and tokens/sec. Increase only after a smoke run. Use fp16 where supported; do not assume bf16 or FlashAttention works on P100.

Qwen's card recommends sampling settings for thinking mode, not greedy decoding. Preserve its documented generation defaults initially; record exact temperature/top_p/top_k and RNG seed. Sample rate and temperature change attack rates, so do not silently tune on final results.

## 5. Install and repository workflow

Recommended: develop on your own computer; Kaggle runs pinned code.

```bash
# locally or in Kaggle terminal after enabling permitted network access
git clone <YOUR_TRACEGUARD_REPOSITORY_URL>
cd traceguard
python -m pip install -e '.[research]'
python -m traceguard.cli doctor
python -m traceguard.cli smoke --config configs/smoke.yaml
```

The angle-bracket URL is a placeholder because a repository has **not** been created. These CLI commands are *planned interface contracts* and are not implemented by this blueprint. Kaggle can disable external network in some competition contexts: if git or package downloads fail, attach a versioned Dataset containing source and wheels instead.

Notebook cells should be thin wrappers, e.g. a checked `subprocess.run([sys.executable, '-m', 'traceguard.cli', 'generate', ...], check=True)`. Store the actual code under `src/`, not in notebooks.

## 6. Qwen trace-extraction validation

Before research trials, generate a few known multi-step tasks and inspect the **raw generated token stream**. Verify:

1. Chat template version and model revision are pinned.
2. `think` segments are genuine output from the open-weight model; not fabricated from a user request or a later rationale prompt.
3. Tool calls and `think` segments can be reliably separated by the chosen tokenizer/parser.
4. Thinking may end early, be truncated, or be malformed; these become explicit status codes.
5. No silent rewriting, summarizing, or stripping of token traces happens between generation and storage.
6. For a monitor asked to decide *before an action*, the view ends at that action proposal and contains no later tool output.

The 2507 version `Qwen3-4B-Thinking-2507` is a different checkpoint that supports **only** thinking mode; do not mix variants without an independent experiment ID. Model card: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 .

## 7. Batch strategy

**Batch 0: CPU correctness** — test state engine, task splitting, generated fixtures, view-projection leakage tests.

**Batch 1: GPU smoke** — 8 episodes, 1 seeded sample, manual read of every generated trace and attempted action.

**Batch 2: pilot generation** — 48 planned episodes, sharded by task family and seed. Persist even partial/invalid episodes and their errors.

**Batch 3: pilot monitoring** — replay exact pilot traces under each evidence condition, storing one monitor score file per condition and input hash.

**Batch 4: additional tasks / main** — only after capability and data-quality gates; choose run sizes based on observed GPU time, not a guessed tokens/sec.

**Batch 5: statistical analysis** — no GPU required for most metrics. Generate preregistered plots before exploring optional subgroups.

## 8. Persistence and crash recovery

```
/kaggle/working/traceguard-output/
  run_manifest.json
  environment.json
  tasks_manifest.json
  completed_episode_ids.txt
  episodes/shard-00001.jsonl
  monitor_scores/condition_A-00001.jsonl
  monitor_scores/condition_C-00001.jsonl
  logs/stderr.log
  artifacts/checksums.sha256
```

Write via atomic temp-file → fsync/close → rename. Periodically snapshot completed JSONL shard files; avoid writing one large mutable result at session end. On restart: validate run manifest and hashes, scan `episode_id`, skip only complete valid records, rerun failed records with the *same* config under an explicit retry policy, and never overwrite prior results invisibly. Save notebook versions and copy outputs into durable Kaggle Dataset/private storage after each run.

Track notebook version ID, pinned package versions, exact model commit SHA, tokenizer revision, random seeds, accelerator type, allocated VRAM, wall-clock time, errors, and output hashes. Include a provenance line for every figure.

## 9. Sample quota budget (measurement-driven)

Assume **nothing** about token throughput before preflight. Measure for 10 diverse episodes and 10 monitor scorings:

- mean/median/p90 generated tokens per episode,
- peak allocated and reserved GPU memory,
- average wall time per episode by task family,
- model download/load time (cold vs warm),
- inference failure/retry rate,
- monitor input token lengths and truncation rate.

Estimate total required hours with observed model timings and a conservative 25–40% contingency, then verify against **your account's displayed weekly quota**. If insufficient, prioritize baseline and paired primary experiment; reduce secondary conditions while clearly documenting it *before* final holdout. A 1% FPR operational claim will require more independent benign data than a small Kaggle pilot can cheaply produce.

## 10. Troubleshooting decision tree

**GPU math crashes:** check compute capability, CUDA/torch wheel compatibility, rerun preflight. If P100-specific, prefer T4 rather than blindly upgrading libraries.

**CUDA OOM:** reduce `max_new_tokens`, prompt history, batch size; use quantization; free stale models; monitor context length. Consider smaller model for pipeline smoke but not as a substitute for viable attack policy.

**Model fails to tool-call:** verify correct chat template/tools schema, inspect raw decoded output, test one deterministic tool example, log parser failures; don't silently coerce invalid actions to a successful rollout.

**Attacks never succeed:** inspect attack attempt vs success, task ambiguity and model capability. If all attacks fail, stop and redesign prior to confirmatory run. Do not change results labels.

**Results lost after disconnect:** restore from saved notebook version/Kaggle dataset, verify checksums, restart only missing episode IDs. Treat the interrupted batch as partial, not as a successful full run.

**High FPR / class imbalance:** recalibrate on an adequately sized development benign set; report operating constraints and uncertainty, not just AUROC.

## 11. Evidence artifact for every GPU session

Deliver a `run_manifest.json`, log, episode shard(s), score shard(s), checksums, and environment snapshot. A run without complete provenance can be used for debugging but must not be presented as a confirmatory result.
