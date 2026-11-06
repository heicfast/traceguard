# UPGRADE v0.7.0 — the AWS era

Branch `upg` (default from v0.7.0). This document records WHAT changed, WHY,
and HOW each change maps onto the program's non-negotiable rules. Every
change is additive to the evidence discipline; nothing weakens a gate.

## 0. Context and motivation

The v0.6 program ran the five-cell monitor matrix on Kaggle T4x2 GPUs under
severe constraints: NF4 quantization for every 14B model, fp16-only Turing
kernels, 8K context caps, ~12 tok/s serial transformers generation, and a
three-account quota fleet. Two scientific limits followed:

1. **Hardware ceiling.** 32B-class monitors, MXFP4 checkpoints (gpt-oss),
   and MoE cross-family monitors were blocked outright (see the registry's
   pre-upgrade `blocked` rows).
2. **Headroom ceiling.** The S5-decisive headroom measurement
   (`runs/v6/headroom_by_family_20261028.txt`) found actions-only rank
   AUROC = 1.000 at the 14B monitor on ALL families and strategies — 0/53
   attacks action-invisible (one-sided 95% upper bound 5.5% < SESOI 0.10).
   On that inventory, a confirmatory marginal-value study is a premeasured
   null. The repo's own predeclared future path: **intent-only attacks plus
   >=40 benign calibration clusters**.

The AWS research-credit budget (~$3–4k; AWS Cloud Credit for Research caps
student awards at $5,000 — verified 2026-11-06) removes constraint 1. This
upgrade removes constraint 2 by design (new intent-era families) and
upgrades every layer — methodology, models, serving, statistics, evidence,
and documentation — to the standard the final paper requires.

## 1. What changed, by subsystem

### 1.1 Execution plane (NEW: `src/traceguard/serving/`)

| Component | What it does |
|---|---|
| `openai_compat.py` | `AgentProvider`-conformant client for any OpenAI-compatible endpoint: locally launched vLLM/SGLang servers (`vllm_openai`) or remote frontier APIs (`api_openai`). Full audit payloads; bounded transport retries; reasoning via `reasoning_content` with think-tag fallback; per-request seeds where supported; `determinism_class` recorded per turn. |
| `vllm_server.py` | Pinned `ServerSpec` → deterministic vLLM argv (revision, dtype, TP degree, max-model-len, prefix caching, reasoning parser, quantization); version floor check (v0.33.x); health-gated launch; process-group teardown; `ServerLaunchRecord` evidence (argv, versions, GPU inventory). MXFP4 models are refused on non-Hopper tiers at ARGV-BUILD time. |
| `cost.py` | Cited `PriceTable` (source URL + verification date mandatory) + append-only NDJSON `CostLedger` with a fail-closed hard budget cap; `cost-report` CLI replays and summarizes. |

Rule 4 (no GPU dependency for core correctness) is preserved: the CPU test
suite (361 tests) exercises the entire serving plane against mock servers.

### 1.2 Model registry (`models/catalog.py`)

- Per-tier admission (`kaggle_t4` / `aws_a100_80gb` / `aws_h100_80gb`);
  nothing transfers across hardware without a tier-matched probe artifact
  (the factory verifies artifact↔tier binding).
- `Verification` record per entry: `verified` (source opened, dated) vs
  `candidate` (probe must confirm repo/API id before research use). This is
  the no-hallucination discipline applied to infrastructure facts.
- New entries: Qwen3-32B (verified), Qwen3.5-14B (candidate), GLM-4.5-Air-FP8
  (candidate, cross-family MoE monitor), gpt-oss-20b unblocked on the H100
  tier (verified: MXFP4 requires Hopper+; A100 stays blocked), and remote
  API monitors (OpenAI GPT-6.1 Sol / GPT-6 Luna verified pricing;
  DeepSeek API verified; Anthropic Opus 4.6 candidate-id).

### 1.3 Task inventory (NEW: `tasks/intent_specs.py`)

Four families with **action-submerged violations**, each grounded in a
verified SLEIGHT-Bench blind-spot category:

| Family | Mechanism | SLEIGHT category | Split |
|---|---|---|---|
| `approval_window` | the violating action is a routine-looking reference update; diagnosis needs a registry JOIN + window comparison | N-hops / system state | development |
| `quota_accumulation` | several individually-policy-clean edits whose SUM crosses a global cap (new engine op `sum_at_most`) | counting / cumulative state | validation |
| `bulk_masking` | one protected rename buried among 5+ legitimate renames | needle-in-haystack | holdout |
| `delegated_authority` | self-approve a ticket, then edit the protected field it guards | authorization confusion | shift |

Properties enforced by tests: initial states satisfy every invariant;
each attack step alone is policy-clean (per-step deniability, quota family);
the full reference attack violates exactly; honest main-goal edits never
violate. Engine bumped to **1.1.0** (additive op; 1.0.2 archives re-grade
identically and stay loadable). Split manifest → `3.0.0-v7` (old assignments
untouched; deviation logged).

### 1.4 Attack-policy axis (`agents/policy.py`)

Four new frozen presentation strategies, grounded in the same taxonomy:
`authorization_claim` (authorization confusion), `incremental_accumulation`
(salami-slicing), `needle_concealment` (bury-in-batch), `n_hop_indirection`
(indirect effect). The base attack instruction (objective, reference action,
ground truth) is unchanged — presentation only.

### 1.5 Monitor instruments (`monitors/followup.py`)

NEW **follow-up interviewing monitor** operationalizing the verified Guan et
al. (arXiv:2512.18311) finding that follow-up questioning plus the agent's
follow-up CoT improves monitorability: monitor asks exactly one
decision-relevant question (frozen prompt, temperature 0) → an independently
pinned elaboration model answers WITH reasoning (captured verbatim) → the
monitor adjudicates the standard v2 rubric over view + labelled exchange.
Composite prompt hash registered in the verification battery.

### 1.6 Statistics (`analyses/`)

| Addition | Purpose |
|---|---|
| `equivalence.py` — TOST on bootstrap replicates | licenses bounded-negative claims ("CoT adds ≤ SESOI") that a nonsignificant superiority test cannot |
| `calibration.py` — Brier, ECE, reliability bins, net benefit (decision curves) | operating-point quality beyond rank metrics |
| `agreement.py` — Cohen's kappa + confusion counts | monitor-ensemble diversity in the matrix |
| `power.py` — Monte-Carlo design simulation | validates the paired-binary approximation against the actual cluster distribution (required by ACADEMIC_METHODS_V6 §5 before freeze) |
| `metrics.py` — Benjamini–Hochberg FDR + cluster-label permutation test | exploratory multiplicity control; exact cluster-level inference |

### 1.7 Paper pipeline + evidence sync

- `paper_export.py` + `traceguard export-paper`: LaTeX (booktabs) + CSV
  tables derived from verified analysis reports only — every row traces to
  a hashed source report; formatting-only, no statistics are computed here.
- `evidence_sync.py` + `traceguard sync-evidence`: S3 mirroring with
  SHA-256 object metadata and verify-after-write re-download; writes a
  `sync_manifest.json` anchored to the run's config hash.
- CLI additions: `serve-model`, `cost-report`, `export-paper`,
  `sync-evidence`, `power-sim`.

### 1.8 Configuration

`AgentConfig`/`MonitorConfig` gain the endpoint fields (`api_base_url`,
`api_model_id`, `api_key_env`, `hardware_tier`) and followup/elaboration
fields; `ExperimentConfig` gains `budget_usd_hard_cap`. The model matrix
(`MatrixModel`) carries the same fields, so matrix cells can mix
transformers, server, and API providers.

## 2. Rules compliance check

| Rule | Status under v0.7 |
|---|---|
| No predetermined conclusion | unchanged — RQ5 (intent-era families) is a hypothesis with predeclared null space; TOST exists precisely to license negatives |
| No fake ground truth | unchanged — the four new families grade by exact state transitions (incl. the new aggregate invariant); the follow-up monitor's elaboration is EVIDENCE, never ground truth |
| No benchmark leakage | unchanged — prompts frozen and hash-bound before evaluation; the followup instrument is registered in the verification battery before first use |
| No GPU dependency for core correctness | strengthened — 361 CPU tests including mock-server integration of the full serving plane |

## 3. What was verified during this upgrade (and how)

- 361 tests pass (was 249); ruff and mypy clean on 69 source files.
- New families: authoring validation + property tests (initial-state
  validity, per-step deniability, sequence-violation, honest-benign).
- Serving plane: end-to-end tests against in-process mock OpenAI-compatible
  servers (reasoning field path, think-tag fallback, greedy request shape,
  seed passing, transport failure, budget-cap fail-closed).
- Statistics: hand-computed fixtures for TOST, Brier, ECE, net benefit,
  kappa, BH (cross-checked against the statsmodels step-up result),
  permutation size, power simulation behavior.
- Registry claims: every `verified` entry's source URL was opened during the
  2026-11-06 research pass (see docs/REFERENCES.md §8); `candidate` entries
  fail closed until a tier-matched probe confirms them.

What still requires the AWS run itself (by design, fail-closed): the
tier-matched admission probes (exact revisions, served-id pinning, VRAM
headroom), and of course the GPU results.

## 4. Upgrade runbook (short form)

1. `scripts/aws/launch_instance.sh p4de.24xlarge <key>` (or `spot`).
2. `git clone` this branch; `pip install -e '.[research,analysis,serving]'`.
3. `traceguard serve-model launch --spec configs/serving/qwen3_14b_a100.yaml`.
4. `python scripts/aws/probe_and_pin.py ...` → probe artifact.
5. `scripts/aws/run_matrix_cell.sh <cell.yaml> <serving.yaml> <run-dir>`.
6. `traceguard export-paper --report .../analysis_report.json --out paper/tables`.

Full procedure: docs/AWS_SETUP.md. Kernel standard: docs/INFERENCE_ENGINEERING.md.
Science: docs/METHODOLOGY_V7.md. Sources: docs/REFERENCES.md.
