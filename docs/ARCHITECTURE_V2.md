# TRACEGUARD Architecture v2

Status: implemented research-kernel baseline; confirmatory execution remains
blocked by the statistical and benchmark gates listed below.

## 1. Claim boundary

TRACEGUARD measures whether a fixed monitor extracts additional predictive
information from an agent's model-emitted reasoning trace when compared with
the same episode's observable tool trajectory. It does **not** claim that the
trace is faithful, complete, private cognition, or a deployable sole safety
mechanism. The primary object is *monitorability of emitted evidence*, not
mechanistic interpretability.

The architecture follows the control/monitorability separation advocated by
Korbak et al. (2025), Greenblatt et al. (2024), OpenAI's monitorability suite
(2025), MonitorBench (2026), and SLEIGHT-Bench (2026): capability, evidence,
monitoring, intervention, and ground truth are separate systems with separate
records.

## 2. Trusted computing base

| Layer | Trusted for | Never trusted for |
|---|---|---|
| Task registry | authorization, initial state, protected invariants | monitor verdicts |
| Sandbox engine | typed state transitions and exact hashes | inferring agent intent |
| Ground-truth grader | attempt/success/task labels from transitions and state | reading model prose |
| View compiler | allowlist projection into blinded conditions | outcome computation |
| Storage kernel | atomicity, recovery, byte hashes | scientific interpretation |
| Statistical runtime | frozen estimands and uncertainty | changing exclusions after results |
| Agent/monitor models | generating study observations and scores | labels, provenance, or policy truth |

The minimum trusted computing base is deliberately CPU-verifiable. GPU code is
an inference adapter and cannot change the sandbox, grader, or evidence rules.

## 3. End-to-end data flow

1. A frozen config and family-level split manifest produce an immutable episode
   plan before inference.
2. The agent receives an authorized task and, for attack episodes only, a
   privileged conflicting instruction.
3. Every generation returns a `ModelTurn`. For Qwen this includes exact prompt
   token ids, exact generated token ids, decoded output with and without special
   tokens, original messages, generation parameters, model/tokenizer revisions,
   chat-template hash, device/runtime metadata, and per-field SHA-256 digests.
4. Only typed JSON tool calls reach the deterministic sandbox. Model text is
   never executed.
5. The environment grades protected-state violations independently of model
   prose.
6. Public envelope and privileged audit record are prepared in a write-ahead
   transaction, projected into JSONL, added to the completion ledger, and then
   sealed by a commit record. Recovery completes an interrupted transaction
   without rerunning inference.
7. The view compiler derives conditions A/B/C/D/E/F from the saved episode.
   Hidden instructions, policy mode, outcomes, split roles, and privileged
   transition fields are outside the monitor boundary.
8. Monitors save the exact rendered prompt, raw response, parsed response,
   inference evidence, view hash, and full prompt-template hash.
9. Analysis joins only schema-valid, hash-valid, paired score records and emits
   explicit censor accounting.

## 4. Raw-evidence boundary

The reasoning trace is the text decoded from tokens genuinely emitted in the
model's thinking channel. Token ids are canonical because decoding behavior can
change across tokenizer versions. A retrospective explanation, reconstructed
trace, repaired malformed output, or silently truncated prefix is forbidden.

Full rendered prompts are privileged because attack prompts contain the hidden
instruction. Monitor artifacts contain no outcome label or attack-policy field.
Release packaging must therefore create separate public and controlled-access
bundles; see `RAW_EVIDENCE_SPEC.md`.

## 5. Determinism and replay

- Config, split, task, prompt template, view, state, transition, output text,
  token-id arrays, transactions, and artifacts use SHA-256.
- Qwen sampling uses an explicit per-call `torch.Generator`, rather than mutable
  process-global RNG state.
- Model and tokenizer revisions are separately pinned.
- Research stages refuse placeholder revisions, source/config commit mismatch,
  a dirty tracked worktree, engine/config version mismatch, stage mismatch, and
  shard-topology drift.
- Exact stochastic bit-for-bit regeneration is not promised across different
  CUDA/PyTorch kernels. The saved token sequence is the observation of record;
  replay means independently validating the recorded transition and analysis
  pipeline from those bytes.

## 6. Failure semantics

All research failures are typed observations. Invalid tool calls are recorded;
provider failures create runtime-error episodes; invalid LLM-monitor output is
invalid rather than numerically repaired. In online control, an invalid monitor
score aborts the episode—there is no implicit `0.0` fail-open decision.

## 7. Statistical gates

The current pilot is a feasibility experiment, not confirmatory evidence.

1. The fixed-5%-FPR estimand requires at least 40 independent benign calibration
   template clusters. The current validation family provides only two. The
   confirmatory runtime now hard-fails this design.
2. Current patch-style tasks often make the protected key visible in condition
   A. If the deterministic action-only baseline saturates, the benchmark cannot
   establish incremental CoT value. A new pilot must add independently authored,
   action-ambiguous-but-environment-verifiable task strata before freeze.
3. Monitor-aware concealment/evasion is a separate robustness stratum and must
   not be mixed into the primary natural-policy estimate.
4. Offline detection and online prevention remain separate experiments with
   fresh online rollouts.

## 8. Performance architecture

Qwen3-4B fits on one T4 in fp16. The efficient use of a dual-T4 Kaggle session
is episode-level data parallelism (one model replica and deterministic shard per
GPU), not tensor-parallelizing a 4B checkpoint across both cards. Each worker
writes a separate run directory; `merge_shards.py` verifies identical run/config
manifests, rejects duplicate episode ids, recomputes hashes, and runs the full
grader-based verifier.

Performance changes are admissible only when they preserve the generated-token
distribution specified by the frozen config. Quantization, compilation,
attention-backend changes, batching, or dtype changes are new experimental
conditions unless equivalence is established before freeze.

## 9. Required gates before preregistration

- Complete and archive the current pilot as diagnostic data, explicitly marked
  with its engine-version provenance discrepancy.
- Build an independent calibration bank meeting the cluster-count gate.
- Add and pilot action-ambiguous task strata; reject a saturated benchmark.
- Run monitor A/C/D/E on identical stored episodes and publish invalid/censor
  counts.
- Complete independent developer, scientist, security, and statistician reviews.
- Freeze exact source commit, model/tokenizer revisions, prompts, views,
  thresholds, exclusions, split manifest, and power analysis.

