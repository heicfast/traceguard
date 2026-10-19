# REVIEW_SYSTEM — Multi-Point-of-View Independent Review

TRACEGUARD enforces independent reviews from four professional POVs at every gate (M4, M6, M7, M8, M9). Reviews are run independently, recorded verbatim under `worklog/reviews/`, and their verdicts gate progression.

## Reviewer roles and mandates

Each reviewer answers ONLY from their mandate; they are instructed to find problems, not to praise. Every finding is classified:

- **BLOCKER** — must be fixed before the gate can pass.
- **MAJOR** — must be fixed before the NEXT gate or explicitly accepted with rationale in `docs/deviations.md`.
- **MINOR** — recorded, fixed opportunistically.

### R1 — Senior software engineer (25+ yrs systems/infrastructure)

Mandate: correctness, determinism, resumability, data integrity, failure handling.
Checklist:
- [ ] Any code path where a failure could be silently swallowed? (`except: pass`, implicit fallback, default value masking errors)
- [ ] Is every persisted record atomic and crash-safe? Is resumption idempotent (no duplicates possible)?
- [ ] Do hashes pin the exact things they claim (state, config, code)?
- [ ] Any nondeterminism that could leak into envelopes (wall times, dict ordering, RNG)?
- [ ] Are typed errors used everywhere instead of stringly-typed checks?
- [ ] Would this code survive a Kaggle session dying mid-shard?

### R2 — Senior research scientist (experimental methodology)

Mandate: does the design actually answer RQ1–RQ4? Are inferences valid?
Checklist:
- [ ] Is the primary estimand exactly the preregistered one? Any result-shopping risk?
- [ ] Is the pairing really within-episode (same trajectory for A and C)?
- [ ] Are clusters handled at the right level (template, not episode)?
- [ ] Is the capability funnel interpretable (attempt/success separation)?
- [ ] Could a null result be published from this design? If not, the design is broken.
- [ ] Are negative-control episodes genuinely exercising FPR paths?
- [ ] Does anything in task design make attacks trivially detectable from actions alone (which would floor ΔR at 0 and prove nothing about CoT)?

### R3 — Senior security researcher (red-team / leakage adversary)

Mandate: adversarial integrity of the evidence boundary and the benchmark itself.
Checklist:
- [ ] Can ANY monitor view be made to reveal policy_mode, outcomes, or hidden instructions (including via ordering, formatting, ids, filenames, lengths)?
- [ ] Can episode ids or view hashes leak labels? (opaque id rule)
- [ ] Could the attack instruction be inferred from the task statement alone (making the hidden objective not actually privileged)?
- [ ] Is the `hidden_instruction` event exclusion airtight across ALL conditions and cutoffs?
- [ ] Are there channels: view length, section presence, event counts, error codes correlated with attack?
- [ ] Could the monitor prompt itself hint at expected conditions (e.g., mentioning reasoning that may not be shown)?
- [ ] Is the noise corpus (condition D) truly content-free w.r.t. attack semantics?

### R4 — Statistician (inference and uncertainty)

Mandate: estimands, thresholds, uncertainty, multiplicity.
Checklist:
- [ ] Is recall-at-calibrated-FPR the right estimand at this sample size, or should the preregistered AUROC fallback apply? Is the precision gate honest?
- [ ] Is the bootstrap resampling at the correct cluster granularity? Are paired conditions resampled jointly?
- [ ] Are degenerate replicates handled and reported (not dropped silently)?
- [ ] Are denominators reported with every rate (no "93%" without n)?
- [ ] Is any multiplicity (multiple conditions/splits) acknowledged with a handling rule?
- [ ] Would the planned CI widths support the claimed conclusions at plausible effect sizes?

## Process

1. At a gate, open four review files: `worklog/reviews/<gate>_<r1..r4>.md`.
2. Each reviewer works INDEPENDENTLY (separate context, no shared notes) against the repository state at the gate's commit.
3. Verdict format per reviewer: findings list with severity + file/line evidence + concrete fix suggestion + PASS/FAIL for the gate.
4. Consolidation: BLOCKERs → fix → re-run affected reviews before gate passes; MAJORs → scheduled with owner; MINORs → backlog.
5. The consolidation record (who found what, what changed, re-review outcome) is committed with the gate tag.

## Feedback loop to implementation

- Every BLOCKER becomes a TODO with a regression test where feasible.
- Fixes re-run the full verification cadence (QUALITY_MANUAL §3).
- Recurring finding classes get a permanent test or check added to CI (e.g., a new leakage test) — the review system is itself iterated on.
