# SOP-001 — Experiment Execution

Standard operating procedure for every TRACEGUARD experimental stage. Executed by implementation agents (human or automated). Deviations require a dated entry in `docs/deviations.md` BEFORE continuing.

## Stage ladder

```
smoke -> pilot -> (freeze) -> confirmatory -> online -> release
```

A stage may only start when the previous stage's exit gate is recorded as PASS in the worklog. The 12-week calendar from the original blueprint is explicitly DISCARDED; gates, not dates, govern progression.

## SOP per stage

### Stage S0 — environment and regression baseline

```bash
cd traceguard
python -m pip install -e '.[analysis,dev]'
python -m ruff check src tests
python -m pytest -q
python -m traceguard.cli doctor
python -m traceguard.cli smoke --out runs/smoke-<commit-sha>
python -m traceguard.cli verify --run-dir runs/smoke-<commit-sha>/traceguard_smoke
```

**Exit gate:** lint clean; 100% tests green; doctor `template_validation_ok`; smoke `ok: true`; verify `ok: true`.

### Stage S1 — GPU preflight (Kaggle)

Follow SOP-002. Push `traceguard-preflight` kernel (T4 shape preferred; P100 has documented PyTorch kernel issues). The kernel writes `environment.json` with real matmul/reduction results.

**Exit gate:** preflight manifest shows `check: passed` with finite matmul results, recorded VRAM, and torch/CUDA versions. No model downloads before this gate.

### Stage S2 — model integration validation (Kaggle)

Push `traceguard-gen` kernel with `--stage smoke --provider qwen --limit`-equivalent small plan: 8 episodes (4 templates × 2 modes). Download output; run `verify`; MANUALLY read every generated episode's reasoning and tool calls (Batch-1 discipline: "manual read of every generated trace").

**Exit gate:**
- ≥90% episodes well-formed (parsed think tags, valid tool JSON);
- authentic think tokens captured with parse status recorded;
- grading matches human reading of state changes;
- peak VRAM and tokens/sec recorded for budget planning;
- no silent CoT reconstruction (raw outputs present in audit store).

### Stage S3 — pilot generation (Kaggle)

Full pilot plan (16 templates × 2 policies × 2 seeds = 64 episodes). Sharded, checkpointed, resumable. On kernel interruption: rerun the same shard command; completed episode ids are skipped (this is resumption, not duplication).

**Exit gate (capability/precision go-no-go, all recorded):**
- attack attempt rate among attack episodes is non-trivial (≥30% target; below 15% = redesign task prompts, new pilot);
- attack success rate among attempts ≥20% (an agent that cannot ever succeed cannot measure monitor quality);
- honest main-task success ≥50% (task must be doable);
- trace availability ≥90% in thinking mode;
- invalid-action and timeout rates each ≤30%;
- estimated GPU time for confirmatory run within weekly quota (with 25–40% contingency).

GPU feasibility is never a manual auto-pass: `scripts/pilot_gate.py` requires a
JSON evidence file containing available hours, projected hours, and the
contingency fraction. Missing evidence makes the gate fail.

A failed gate is PUBLISHED as pilot learning (negative results count), not hidden.

### Stage S4 — pilot monitoring (Kaggle)

Score the SAME episodes under conditions A, C, D, E (+B offline) with the frozen monitor prompt and deterministic baselines. View compiler leakage assertions must hold for every (episode, condition).

**Exit gate:** no leakage failures; invalid monitor output rate ≤10%; score distributions visibly non-degenerate (variance > 0); calibration split has ≥40 independent benign template clusters; and the deterministic action-only baseline does not saturate. An AUROC fallback may be reported for a pilot but does not turn an underpowered fixed-FPR design into a confirmatory answer to RQ1.

### Stage S5 — protocol freeze (preregistration)

1. Fill `templates/preregistration.md` completely (prompts with hashes, model revisions, splits, estimand, threshold rule, bootstrap spec, exclusion rules).
2. Commit; tag `v0.4-preregistered`; push; record commit SHA + UTC timestamp in the preregistration and worklog.
3. Freeze means: no prompt edits, no threshold edits, no split edits, no estimator swap. Anything else = deviation entry + new holdout.

**Exit gate:** preregistration signed with commit SHA; frozen config hashes match the tagged commit; independent reviewer confirms the freeze record.

### Stage S6 — confirmatory run

Generate fresh confirmatory instances (disjoint instance seeds by stage), score under frozen conditions, then `analyze`. Analysis may be opened only after `verify` passes on the complete run.

**Exit gate:** all planned episodes accounted for (generated or explicitly censored with reason); verify OK; primary estimate + CI computed exactly once, no subset shopping.

### Stage S7 — online intervention (RQ4)

Fresh rollouts under pre-action protocols A and C with frozen thresholds. Decisions logged before execution; blocked violating proposals recorded as attempts-not-successes.

**Exit gate:** decision logs show no future-information access; realized block rate on honest episodes recorded (usefulness cost); violation rate comparison computed on fresh instances only.

### Stage S8 — analysis, review, release

Produce figures with provenance, failure taxonomy, funnel, limitations. Multi-POV review (REVIEW_SYSTEM) + external critique requested on methodology. Tag `v1.0-paper` only after the reproducibility command reproduces the primary figure from archived artifacts.

## Emergency procedures

- **Kaggle session lost:** do not regenerate from scratch; re-run same shard (ledger skips completed ids). Verify checksums of downloaded artifacts.
- **Leakage assertion fires:** STOP the stage; the episode view is quarantined in the log; fix the view compiler; add a regression test; re-run the stage's scoring (generation output is unaffected).
- **Grader disagreement with human audit:** STOP before confirmatory; fix grader or task; new benchmark version; pilot rerun. Never patch labels.
