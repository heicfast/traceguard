# RQ4 online study — attempt-1 (externally cancelled at 110/128 episodes)

**Kernel:** `umar0x/traceguard-online-study` v1, pin `b0e84cf` (review-hardened),
frozen threshold 0.16 (validation-split calibrated; see preregistration).

**What happened:** All 10 validation gates passed (model load, authentic think
tokens, VRAM release, dataset-identity, agent-identity pin-normalized). Both
per-GPU workers then generated online rollouts cleanly — 49 (gpu-00, shard 0)
+ 61 (gpu-01, shard 1) = **110 of 128 episodes completed with zero recorded
worker failures** in ~5.5 h — before the run was cancelled externally (Kaggle
status `CANCEL_ACKNOWLEDGED`, during the 2026-10-28 recovery session; the
18 remaining episodes were never started).

**Decision (recorded in docs/deviations.md):** the online study re-runs in
full with the UNCHANGED kernel (v2 push). The alternative — seeding a resume
from these 110 episodes (the runtime natively skips ledger-complete episodes,
same run_id/config hash) — was considered and rejected: the online study is
not the session's critical path (the cross-family monitor cell is), so the
~4.5 h saving buys no schedule, while a kernel edit re-opens the config-pin
substitution surface where the v1/v2 gate bugs lived (owner directive:
eliminate error/re-fix loops). Per the frozen no-partial-mixing policy these
episodes contribute nothing to the research record; this archive is the audit
trail. The 110/110 clean generation rate is itself evidence the online
machinery (shared-checkpoint rollouts, pre-action gate, WAL ledger) works at
cohort scale on T4 hardware.

**Layout:** `gpu-00/` + `gpu-01/` (worker run dirs: episodes, transactions,
ledgers, decisions in privileged records), `environment.json` (10/10 gates).
