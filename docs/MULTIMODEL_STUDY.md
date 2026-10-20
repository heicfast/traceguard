# Multi-model study and T4 execution standard

Status: implemented architecture; model revisions and probe hashes are not yet
frozen. This document governs the next pilot. It does not retroactively change
the completed 64-episode result.

## 1. Scientific objective

The study estimates a two-sided property of an **agent/monitor pair**. A larger
agent can produce more capable or more legible reasoning; a larger monitor can
extract more signal; quantization can alter either. Consequently, "try a bigger
model" is not one treatment. TRACEGUARD separates:

1. agent scale at a fixed monitor;
2. monitor scale at a fixed agent and identical stored trajectories;
3. agent-family transfer;
4. monitor-family transfer;
5. quantization sensitivity;
6. zero-shot versus development-only QLoRA monitoring.

The primary anchor remains Qwen3-4B agent × Qwen3-4B monitor for continuity.
All other cells are predeclared robustness or mechanism tests. Results are
reported per cell before any synthesis. A pooled average may not replace a
failed or sign-reversed cell.

This directly follows the agent/monitor distinction and inference-compute
findings in OpenAI's *Evaluating chain-of-thought monitorability* suite, while
retaining TRACEGUARD's executed state-machine ground truth and within-episode
evidence controls:

- <https://openai.com/index/evaluating-chain-of-thought-monitorability/>
- <https://arxiv.org/abs/2512.18311>
- <https://arxiv.org/abs/2507.11473>
- <https://alignment.anthropic.com/2026/sleight-bench/>
- <https://arxiv.org/abs/2603.28590>

## 2. Correct T4 feasibility boundary

Three Kaggle accounts increase independent compute quota, not per-process VRAM.
Each two-T4 session still has two separate approximately 16 GB devices without
an assumption of NVLink or pooled memory. The preferred design is one model per
GPU or episode-level data parallelism. Tensor parallelism is a separately
benchmarked backend treatment, never an invisible optimization.

Q4_K_M is a GGUF/llama.cpp quantization label. TRACEGUARD's Transformers path
uses bitsandbytes NF4 or INT8 and records the exact quantization configuration.
These formats are not interchangeable experimental conditions.

Standard FlashAttention-2 CUDA supports Ampere, Ada, and Hopper. A Tesla T4 is
Turing (sm_75), so the default remains PyTorch SDPA with fp16. No kernel may
silently claim FlashAttention-2 on T4. The upstream compatibility statement is:
<https://github.com/Dao-AILab/flash-attention>.

The official GPT-OSS Transformers guide says MXFP4 requires Hopper or newer and
the bf16 20B path requires about 48 GB. Therefore GPT-OSS-20B is **blocked from
the canonical T4 matrix**. A community GGUF port would be a separate checkpoint
and backend treatment, not "the same model on T4":
<https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers>.

Advertised context is a model capability ceiling, not a safe allocation. The
runtime enforces `prompt_tokens + max_new_tokens <= max_context_tokens` and the
admission probe records peak allocated/reserved VRAM. Initial T4 cells use an
8,192-token cap. A higher cap requires a new probe artifact and is analyzed as
a context-budget treatment.

## 3. Audited candidate registry

The executable registry is `traceguard.models.catalog`. It rejects unknown
models and hardware-blocked combinations.

| Checkpoint | Intended role | T4 path | Admission status |
|---|---|---|---|
| `Qwen/Qwen3-4B` | anchor agent/monitor | fp16, SDPA | validated path |
| `Qwen/Qwen3-14B` | scaled agent or monitor | bitsandbytes NF4, fp16 compute | probe required |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | transfer agent/monitor | NF4, tagged reasoning parser | probe required |
| `ServiceNow-AI/Apriel-1.6-15b-Thinker` | exploratory transfer | NF4; template/parser/tool probe | probe required |
| `google/gemma-4-12B-it` | cross-family monitor/fact checker | NF4 text monitor | probe and weight access required |
| `openai/gpt-oss-20b` | future non-T4 monitor | official MXFP4 | blocked on T4 |
| `Qwen/Qwen3.8-27B` | future backend study | separately validated low-bit runtime | blocked in canonical Transformers/T4 arm |

Primary model cards:

- <https://huggingface.co/Qwen/Qwen3-14B>
- <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>
- <https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker>
- <https://huggingface.co/google/gemma-4-12B-it>
- <https://huggingface.co/openai/gpt-oss-20b>
- <https://huggingface.co/Qwen/Qwen3.8-27B>

Registry inclusion does not imply a passed probe. Exact model and tokenizer
commits, licenses/access state, template hash, parser behavior, JSON/tool-call
conformance, authentic trace capture, throughput, and VRAM headroom must be
captured before a cell becomes runnable.

## 4. Admission and freeze sequence

For every probe-required role:

1. create a candidate config with exact model/tokenizer revisions;
2. run `traceguard probe-model --config ... --role agent|monitor --out ...` on
   the target Kaggle T4 image;
3. retain the complete generation audit and runtime/package snapshot;
4. require every gate to pass, including fp16/SDPA, JSON conformance, trace
   channel for reasoning agents, and the declared VRAM budget;
5. copy the artifact SHA-256 into the matrix;
6. freeze `configs/model_matrix.template.yaml` as a versioned study matrix,
   run `traceguard matrix-plan`, then `traceguard matrix-materialize` to emit
   immutable generation/monitor configs and the account schedule;
7. do not update packages, revisions, context limits, quantization, prompts, or
   adapters after the freeze.

At execution, mount the retained JSON artifact and set
`TRACEGUARD_AGENT_PROBE_ARTIFACT`, `TRACEGUARD_MONITOR_PROBE_ARTIFACT`, or
`TRACEGUARD_FACT_CHECKER_PROBE_ARTIFACT` as appropriate. The provider factory
recomputes the artifact hash, checks model/role identity and every recorded
gate, and refuses to load on a mismatch.

The matrix schedule deduplicates generation cohorts. Multiple monitors score
copies of the **same stored episodes**, preserving the paired monitor-capacity
comparison. Independent generation replicates receive different predeclared
replicate IDs; they are not pseudo-replicates of one trajectory set.

`scripts/run_gpu_workers.py` places one agent copy on each T4 and partitions
episodes. `scripts/run_monitor_workers.py` does the same for monitoring: each
worker receives a private run-directory copy, scores a disjoint deterministic
episode shard, then the parent schema-validates and conflict-checks score IDs
before an atomic merge. Workers never write the same JSONL file concurrently.

## 5. Proposed first matrix

The minimum informative matrix is:

| Cell | Agent | Monitor | Question |
|---|---|---|---|
| primary anchor | Qwen3-4B fp16 | Qwen3-4B fp16 | continuity |
| agent scale | Qwen3-14B NF4 | Qwen3-4B fp16 | does stronger reasoning change signal? |
| monitor scale | same Qwen3-4B episodes | Qwen3-14B NF4 | was the pilot monitor underpowered? |
| agent transfer | DeepSeek-R1-Distill-Qwen-14B NF4 | Qwen3-4B fp16 | does result transfer across post-training? |
| monitor transfer | same Qwen3-4B episodes | Gemma-4-12B NF4 | does result transfer across monitor family? |

Two generation replicates are planned per cohort in the template. Episode
counts are chosen only after the 128-episode pipeline pilot supplies attack
attempt, trace-coverage, invalid-score, and intracluster-correlation estimates.
The previous rough 550–650 total is not multiplied blindly across cells.

## 6. Fine-tuning policy

Fine-tuning is not automatically "better" evidence. Training a monitor on the
evaluation families would leak labels and inflate the claimed result. The
zero-shot frozen monitor remains primary.

The optional QLoRA arm is secondary and follows these rules:

- `traceguard prepare-tuning` admits **development split only**;
- validation remains threshold calibration; holdout and shift never create
  gradients or checkpoints;
- targets come from the deterministic state-machine outcome, contain no
  generated rationale, and retain source-view hashes;
- the training corpus, base revision, seed, hyperparameters, package versions,
  checkpoints, and every adapter file are hashed;
- NF4 with double quantization and fp16 compute is the T4 recipe;
- LoRA targets all linear layers, following PEFT's QLoRA guidance;
- a tuned adapter is frozen to an immutable revision before any validation,
  holdout, or shift scoring;
- zero-shot and tuned arms are never pooled.

Implementation: `scripts/train_monitor_qlora.py`. Primary documentation:

- <https://huggingface.co/docs/transformers/quantization/bitsandbytes>
- <https://huggingface.co/docs/peft/developer_guides/quantization>
- <https://huggingface.co/docs/peft/package_reference/lora>

The deterministic training targets initially have only three outcome-derived
levels. Therefore the tuned arm must still pass the score-resolution gate; a
failure is reported, not repaired after seeing holdout results.

## 7. Evidence and reporting additions

Every model turn now retains:

- exact model, tokenizer, and optional adapter revisions;
- audited model-profile hash and admission-probe hash;
- quantization configuration, compute dtype, attention backend, device map,
  context cap, model memory footprint, and per-turn peak allocated/reserved VRAM;
- rendered prompt/messages, raw decoded output with and without special tokens,
  complete prompt/generated token IDs, generation parameters, effective seed,
  parse status, timings, and cryptographic hashes.

Every run manifest retains a secret-free full package inventory, safe runtime
environment, CUDA/PyTorch flags, GPU identity/driver query, model profile, and
snapshot hash. The environment collector uses an allowlist and cannot capture
Hub tokens or other credentials.

Analyses include trace coverage overall and by split. CoT contrasts are blocked
below the frozen 90% trace-coverage threshold. Reports identify the agent,
monitor, revisions, matrix cell, replicate, quantization, and adapter. Matrix
results must show cell-level estimates, uncertainty, censoring, capability
funnels, and gates before any across-cell summary.

## 8. Interpretation discipline

- A Qwen3-14B gain over Qwen3-4B is not automatically a parameter-scale effect
  because NF4 differs from fp16; it is a system-level cell comparison until a
  quantization-matched sensitivity arm exists.
- DeepSeek-R1-Distill-Qwen-14B is organization/post-training diversity but is
  Qwen-derived; it does not alone establish architecture-family independence.
- A stronger monitor recovering CoT value supports monitor-capacity failure in
  the pilot. It does not invalidate the pilot observation.
- More context is useful only when the evidence requires it. Unused context
  raises memory, latency, truncation, and distraction risk without improving
  identification.
- Raw reasoning is untrusted telemetry, not privileged ground truth and not a
  sole production safeguard.
