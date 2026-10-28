# Cross-family monitor cell — attempt-4 partial output (externally cancelled)

**Kernel:** `heicfast/traceguard-mon-cross-deepseek14` (batch), pin `63b43cb`
**What happened:** The run passed every kernel gate and the single_split worker
scored cleanly with zero OOM events — rules_v1 4/4 conditions, keywords_v1 4/4
conditions, and llm (DeepSeek-R1-Distill-Qwen-14B, NF4, split across both T4s)
3/5 conditions complete at 128/128 scores each (`actions`, `actions_plus_cot`,
`actions_plus_noise`) — when the run was cancelled externally (Kaggle status
`CANCEL_ACKNOWLEDGED`). Remaining at cancellation: `actions_plus_cot_late_fusion`,
`actions_plus_shuffled_cot`. No `environment.json` was persisted (kernel chain
did not finish).

**Decision (frozen policy, same as attempt-3):** no partial scores mix into the
research record. This directory preserves the partial output for the audit trail
only. The cell is re-run in full (attempt-5, hosted on `heicjpg` because the
`heicfast` account's remaining GPU quota (5.87 h) is below the measured full-run
requirement; see docs/deviations.md 2026-10-28 entry).

**Layout:** `traceguard-mon-cross-deepseek14.log` (kernel log, truncated at
cancel), `worker_run/` (the worker's projected run with the partial scores).
