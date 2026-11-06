# Inference engineering standard (v0.7.0)

The kernel and serving standard behind the AWS execution tier. Every choice
here is a RECORDED TREATMENT in the study's provenance, never an invisible
optimization. Sources: docs/REFERENCES.md §6 (all opened during the
2026-11-06 pass).

## 1. Engine choice

- **vLLM v0.33.x** is the reference engine (pin `vllm>=0.33.0,<0.34`).
  The `VLLMServerManager` enforces the version floor and records the full
  argv + versions + GPU inventory as evidence (`ServerLaunchRecord`).
- **SGLang v0.5.22** is an accepted alternative backend for prefix-heavy
  monitor workloads (RadixAttention; arXiv:2312.07104); if used, record it
  as a separate backend treatment — never pool across engines silently.

Why not transformers-serial anymore: the T4 program measured ~12 tok/s
single-stream generation; vLLM's continuous batching + PagedAttention
(SOSP 2023) turns the same A100 into hundreds to thousands of aggregate
tok/s across the episode shard, which is the difference between a study
that fits the grant and one that does not.

## 2. Attention kernels, by tier

| Tier | GPUs | Kernel path |
|---|---|---|
| `aws_h100_80gb` (p5) | H100 (sm90) | Hopper-native async + low-precision attention (FlashAttention-3 class, arXiv:2407.08608; up to ~1.5-2x FA2, FP8 support). vLLM V1 selects it automatically; the launch record captures the effective backend. |
| `aws_a100_80gb` (p4de) | A100 (sm80) | FA2/FlashInfer-class kernels — FA3's Hopper-specific wgmma path does not apply; do not claim it. |
| `kaggle_t4` (legacy) | T4 (sm75) | PyTorch SDPA fp16 (the validated v0.5/v0.6 path, unchanged). |

Rule: the attention backend is recorded per launch; a paper table never
compares cells across silently different kernels.

## 3. DeepSeek's kernel line and where it applies to us

The DeepSeek open-source kernel line — **FlashMLA** (multi-head latent
attention kernels), **DeepGEMM** (FP8/FP4/BF16 GEMM + fused MoE, DeepJIT,
SM90/SM100, up to ~1550 TFLOPS on H800), **3FS** (distributed storage),
**DualPipe** (bidirectional pipeline parallelism) — powers DeepSeek-V3/V4
class architectures. TRACEGUARD's dense open-weight monitors (Qwen3 dense
line) do not use MLA, so these kernels act indirectly: through vLLM's
support for MLA models (relevant if a DeepSeek-architecture monitor is
admitted later) and through the FP8 serving conventions DeepGEMM
established. When a DeepSeek-family MoE monitor enters the matrix, its
serving spec MUST use the engine's MLA-aware path and record it.

## 4. Quantization policy (study factor, not a free speedup)

1. **bf16 is the reference arm** on A100-80/H100 for every model that fits.
2. **Official FP8 checkpoints only** (e.g. Qwen3-32B-FP8, GLM-4.5-Air-FP8):
   quantization is a RECORDED factor; FP8 cells never pool with bf16 cells
   (MULTIMODEL_STUDY §8 discipline, carried into v0.7).
3. **MXFP4 (gpt-oss)**: Hopper-or-newer only (verified against the vLLM
   gpt-oss announcement); the argv builder refuses it on Ampere.
4. **bitsandbytes NF4** remains the legacy T4 recipe, admitted only for
   continuity cells.

## 5. Determinism, seeds, and reproducibility classes

| Class | Meaning | Evidence |
|---|---|---|
| `seeded_local` | transformers + fork_rng seed isolation (gold standard; the v0.5/v0.6 path) | full token ids |
| `seeded_server_best_effort` | vLLM per-request `seed`; greedy monitor scoring is stable in practice, but bit-exactness across server versions is NOT guaranteed | full raw outputs + messages + usage; token ids via server only if exposed |
| `unseeded_api` | remote frontier APIs; sampling not reproducible | full raw outputs + messages + usage; `token_ids_complete_when_provider_supports_them` = false |

The evidence contract (RAW_EVIDENCE_SPEC) records the class per turn.
Reproducibility of ANALYSIS is anchored in retained raw evidence, not in
bit-replay.

Additional flags:
- `--enable-prefix-caching`: pure performance for the shared monitor
  prefixes; output-invariant. Recorded in the spec.
- Speculative decoding (EAGLE-3 class) is OFF by default: it changes the
  sampling path. A speculative arm, if ever used, is its own recorded
  treatment.
- `--disable-log-requests`: prompts are privileged evidence and never
  belong in server logs.

## 6. Context budgets

`max_model_len` is pinned by the serving spec to the STUDY's context budget
(32,768 for monitors; 16,384 for agents in the v0.7 template) — never the
model's advertised maximum. The reasoning evidence window doubles vs the
T4 program (12,288 → 24,576 chars) because A100-80 KV headroom allows it;
the window stays a predeclared, config-frozen quantity applied identically
to every reasoning-bearing condition.

## 7. Throughput planning

Plan by tokens, not vibes: `traceguard study-plan` emits generation and
monitoring token estimates from the frozen config; divide by the measured
server throughput (the probe's test generation + a short warmup batch
give a first estimate; the first landed cell gives the real one). Record
`compute_hours` into the cost ledger (`traceguard cost-report`) so the
grant accounting is part of the evidence chain.

## 8. Change control

Any change to a serving spec (flags, dtype, TP degree, context, engine
version) is a NEW `ServerSpec` hash → a new treatment. Never edit a spec
in place after a cell has run; add a new dated spec file and a deviation
note. This mirrors the config-freeze discipline that governs everything
else in the program.
