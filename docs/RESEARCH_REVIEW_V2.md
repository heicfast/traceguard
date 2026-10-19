# v2 Independent Review Consolidation

Date: 2026-10-19. Scope: repository through `origin/main` at `87f7ef6`, original
blueprint, attached Korbak et al. paper, archived GPU smoke evidence, and the v2
changes on this branch. This is a consolidation; formal gate sign-off still
requires separately authored reviewer files after the new pilot.

## Senior systems engineer

- **BLOCKER fixed:** public and privileged JSONL writes were not atomic as a
  pair. A crash between them could cause an orphan or duplicate on resume. v2
  adds write-ahead transactions, idempotent projection, commit markers and
  recovery without inference reruns.
- **BLOCKER fixed:** checksum code hashed a binary file's path string instead
  of its bytes. All artifacts now use byte SHA-256.
- **MAJOR fixed:** strict mypy failed in eight locations; CI now treats typing
  as a gate.
- **MAJOR fixed:** generation previously used only one GPU per dual-T4 session.
  v2 now launches one isolated replica per visible GPU with disjoint global
  shards, separate logs/run directories, and a verified post-run merge.
- **MAJOR open:** the Kaggle bootstrap code remains duplicated across kernel
  entry points and should be consolidated after this pilot without changing the
  frozen inference path.

## Senior research scientist

- **BLOCKER open:** the current validation split has two independent template
  clusters; the protocol requires forty for the fixed-5%-FPR claim. v2 blocks
  confirmatory generation until the requirement is met.
- **BLOCKER open:** most attacks expose protected key/value semantics directly
  in actions. A saturated action-only baseline would make the study unable to
  estimate meaningful incremental CoT value. Add an independently authored
  action-ambiguous stratum and rerun the pilot.
- **MAJOR open:** add monitor-aware concealment as a separately powered
  robustness experiment, informed by MonitorBench and SLEIGHT-Bench, without
  contaminating the natural-policy primary estimate.
- **MAJOR fixed:** research analysis could silently fall back from validation
  to development calibration. That fallback is now smoke-only; research stages
  fail explicitly.

## Senior security researcher

- **BLOCKER fixed:** monitor non-JSON numeric recovery remained `valid=True`;
  online control could therefore fail open at score 0. v2 marks recovered
  outputs invalid and aborts online episodes on invalid scores.
- **MAJOR fixed:** exact model prompts, hidden instructions, token ids and raw
  output are segregated into privileged evidence; monitor views remain
  allowlist-compiled.
- **MAJOR open:** add property-based delimiter/prompt-injection fuzzing and a
  release-time scan proving that no privileged field enters public score/view
  bundles.

## Statistician

- **BLOCKER open:** four holdout template clusters are still too few for stable
  cluster-bootstrap tail claims. A prospective power/simulation study must set
  the final number of independently authored templates before freeze.
- **MAJOR fixed:** confirmatory execution now checks the maximum possible benign
  calibration clusters before spending GPU quota.
- **MAJOR open:** report paired episode-level contrasts alongside cluster-level
  uncertainty, all denominators, degenerate bootstrap counts, realized holdout
  FPR, and sensitivity to cluster definition. Do not interpret AUROC fallback
  as answering the fixed-FPR primary question.

## Consolidated verdict

**Architecture v2: suitable for a new pilot after merge. Confirmatory study:
FAIL/BLOCKED.** The code-level evidence defects are repaired, but the task bank,
calibration-cluster count, action-only saturation check, and prospective power
analysis must pass before preregistration.

