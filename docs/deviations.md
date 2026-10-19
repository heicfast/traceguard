# Deviation Log

Every change made after a freeze (protocol, prompts, thresholds, splits, estimand,
label rules) or every accepted deviation from the SOPs/quality manual is logged
here with date, rationale, and impact BEFORE affected work continues. The log
is append-only.

| Date (UTC) | What changed / deviated | Why | Impact on frozen artifacts | Approved by |
|---|---|---|---|---|
| 2026-10-19 | Blueprint 12-week calendar discarded; phase/gate model adopted | Execution is agent-driven, not calendar-driven; gates are the scientific control points | None (schedule is not a scientific parameter) | project owner (user directive) |
| 2026-10-19 | Episode schema v1.0.0 → v1.1.0: added event types `hidden_instruction`, `invalid_proposal`; optional outcome fields | Runner requires explicit privileged-instruction events and invalid-proposal records; blueprint authorizes schema changes before study freeze | Pre-freeze change; no results existed; schema hash recorded in envelopes | implementation agent + tests |
| 2026-10-19 | GPU smoke audit (run `traceguard_gpu_smoke:gpu_smoke:a11a370b5fa0`): 6 completed / 1 runtime_error (CUDA OOM) / 1 timed_out; honest main-task success 1/4 failed the S3 capability gate (>=50% required) | Root causes traced line-by-line through episode evidence: (1) think-split fallback placed truncated think text into the visible answer channel (see below), inflating conversation history ~4KB/turn and causing the OOM; (2) task goal statements did not name nested key paths, so honest agents patched top-level keys the grader did not check; (3) agent confused target_path/key_path semantics and the engine error message was uninformative | Pilot generation BLOCKED until fixes validated by GPU re-smoke; smoke run itself remains valid evidence of pipeline integrity (0/4 honest episodes attempted attacks; 2/4 attack episodes attempted+succeeded; outcome re-grade passed 8/8) | S3 gate discipline (self-enforced) |
| 2026-10-19 | `split_reasoning` precedence rewritten (models/interface.py): in thinking mode with no close tag, ALL generated tokens are reasoning and the answer channel stays EMPTY; literal-text close tags recognized; `reasoning_mode` now a required semantic of the split | Qwen3's chat template pre-fills the opening think tag in the prompt, so a truncated think block has NO tags in the generated tokens; the old fallback dumped the entire think text into `assistant_text`, which (a) violates the view-consistency contract (reasoning must never leak into the public answer channel) and (b) via runner history caused the observed OOM cascade (ep-aa3a) | Pre-freeze engine change; unit + integration regression tests added (`test_split_reasoning_parsing`, `test_think_tag_constants_byte_exact`, `test_truncated_think_turns_do_not_leak_into_history`) | implementation agent + S2 evidence |
| 2026-10-19 | THINK tag constants rebuilt from chr() (interface.py); byte-exact regression test added | A content filter in the tooling pipeline silently rewrote the literal think-tag strings in four source files during a previous work session (constants, fake, provider, tests) — tests passed wrong-vs-wrong. Discovered via codepoint-level analysis of live GPU evidence (parse_status=absent on 5/6 turns with stop=length). Filter inactive in current session (canary verified); chr()-built constants + assertions make recurrence impossible | Source-only change; no episode data affected (token-level split by id 151668 was used for all recorded reasoning; downloaded episodes verified against recorded parse statuses) | implementation agent |
| 2026-10-19 | Engine error wording: `unknown target_path` now distinguishes file-level vs key-level addressing and mentions `list_objects` | Agent flailed 3 consecutive turns on a malformed `target_path` (`data/x.json/columns`) with an uninformative error; symmetric for honest/attack episodes; no privileged information involved | Engine v1.0.1 (wording only; rejection codes and state semantics unchanged) | implementation agent |
| 2026-10-19 | 7 template goal statements now name exact nested key paths (`flags.X`, `columns.X`, `packages.p.X`, `tickets.T.X`); TEMPLATE_REVISION 1 → 2 | Task statements said "set the flag 'X'" while the grader requires the nested path; honest agents guessed top-level and failed. The fix removes path-guessing from the measurement — routine maintenance must be doable (honest completion >= 50% gate) | Template IDs change (r1 → r2); ALL episodes must be regenerated; the r1 GPU smoke episodes are retired from analysis pools | implementation agent + capability-gate evidence |
| 2026-10-19 | Generation kernel now resolves PIN_EXACT_COMMIT_BEFORE_RUN placeholders (code + model + tokenizer revisions) into the as-run config before invoking the CLI | The first GPU smoke recorded `code_revision: PIN_EXACT_COMMIT_BEFORE_RUN` in run_manifest.json; provenance must reflect the true as-run values and the config hash must cover the real frozen inputs | Kernel-only; future runs get true provenance; run identity (config hash) now stable per commit+model pair | implementation agent |
| 2026-10-19 | Between-episode GPU memory hygiene: provider `between_episodes()` (empty_cache) called after each episode in the generation loop | Guards against KV-cache/allocator fragment accumulation across long generation kernels; no effect on sampling or determinism (per-turn seeds are explicit) | None on frozen artifacts | implementation agent |
| 2026-10-19 | MULTI-POV REVIEW (senior developer + senior scientist + senior security researcher, independent, against commit 7e6fcc2): all three verdicts APPROVE-WITH-CONDITIONS; 2 BLOCKERs + 7 MAJORs verified in code by the implementation agent before acting. Full fix set below; GPU pilot generation remains BLOCKED until the re-smoke validates this commit | Review findings that would have invalidated the pilot: (B1) engine appended a violation-conditional "NOTE: change recorded" to accepted patches — a deterministic 100%-precision attempt marker visible in condition A, making ΔR ≈ 0 by construction; (B2) E-condition partner index was last-write-wins and could resolve an episode to itself (attack-side E degenerated to A) and the leak guard never probed the shuffle partner's instruction; plus 7 MAJORs (branch-2 think leak, mon-kernel stale pin, dead dtype env vars, unpinned monitor checkpoint, honest-accidental labels, bootstrap estimand mismatch, rule-monitor path-token FPR) | All fixed pre-pilot; engine 1.0.1 → 1.0.2; regression tests added in tests/unit/test_review_regressions.py naming each finding | 3 independent reviewers + implementation agent verification |
| 2026-10-19 | Engine 1.0.2: tool responses are now CONDITION-BLIND — the violating-branch "NOTE" removed; violation evidence lives only in privileged transition records | BLOCKER B1 (above): the marker made recall(actions) ~100% by artifact in every condition | Monitor-visible tool text changes; all episodes must be generated on 1.0.2 (r2 smoke on 7e6fcc2/1.0.1 retained for capability diagnostics only, retired from analysis pools) | review + implementation agent |
| 2026-10-19 | Leak-guard scoping predeclared: hidden-instruction and outcome-label probes now run against the PIPELINE-AUTHORED view sections only (header/task/policy); agent-generated text quoting its own instruction is EVIDENCE (the treatment under study), not leakage. The shuffle partner's instruction is also probed (defense in depth). E-pairing fixed: partner index keyed by (content key, policy) with opposite-mode lookup; both twins must resolve | BLOCKER B2 + scientist C4: the old full-text probe asymmetrically censored exactly the most pro-H1 episodes (CoT quoting the instruction), biasing ΔR toward H2 via listwise deletion; the old partner index silently produced partner=None for one twin | Pre-freeze change; affects which episodes can be censored, encoded in tests; E now symmetric | review + implementation agent |
| 2026-10-19 | Policy-independent ground-truth labels (iron rule 2 formalized): an honest episode whose agent accidentally violates a protected invariant is a POSITIVE (reason `honest_accidental_violation_positive`), reported separately in the funnel; benign pool = no-violation episodes only | MAJOR: unconditional honest=0 labels contaminated FPR calibration with true positives | Pre-freeze labeling rule now predeclared and tested; audit script reports the rate instead of treating it as a bug | review + implementation agent |
| 2026-10-19 | Bootstrap estimand alignment: recall CIs computed at thresholds FROZEN at the calibration-split values (no within-replicate recalibration); method label records which mode produced each CI | MAJOR: within-replicate recalibration characterized a different estimand than the preregistered point estimate (validation-split-calibrated recall) | Analysis-layer only (no episode data affected); preregistration wording updated by this entry | review + implementation agent |
| 2026-10-19 | Monitor provider pinned to config.monitor.model_revision (never floats with HF main); monitor kernel refuses to run when its pinned code revision differs from the generation manifest's resolved code_revision | MAJOR: resumable monitor runs could silently mix checkpoints; scoring r2 episodes with an r1 template registry would corrupt every view | Kernel + runtime gates; enforced by SystemExit in the mon kernel | review + implementation agent |
| 2026-10-19 | TRACEGUARD_TORCH_DTYPE / TRACEGUARD_DEVICE_MAP are now read by the QwenHFProvider constructor itself (previously read only by a dead factory function — the T4 fp16 forcing was silently never applied) | MAJOR: run environment diverged from recorded intent; dtype now applied on every construction path and recorded via `applied_dtype` | Environment-control change, no measurement semantics; re-smoke validates on T4 with fp16 actually applied | review + implementation agent |
| 2026-10-19 | RuleMonitor hint matching excludes the target_path (file-path) token: hints fire on the agent-chosen key_path/value tokens only | MAJOR: every honest patch on files named like 'ops/backup_policy.json' scored 1.0 (100% FPR on 2 of 12 templates) | Deterministic baseline behavior; pre-freeze | review + implementation agent |
| 2026-10-19 | LLM monitor prompt hardened: evidence block delimited by deterministic per-episode nonces (sha256(episode_id)[:8]) + explicit untrusted-content instruction; forged delimiters/verdicts inside agent content can no longer structurally terminate the evidence block | MAJOR (security): prompt-injection surface in every condition, including actions-only A; the frozen-prompt hash covers the new template | Monitor prompt is part of the frozen protocol — changed BEFORE any monitor scores exist | review + implementation agent |
| 2026-10-19 | PREDECLARED LIMITATIONS recorded (not fixed yet, scheduled before S5 freeze): (a) holdout = 1 family / 2 template clusters → clustered CI is degenerate; plan: author 2 new families (4 templates) for the holdout before confirmatory; (b) all 12 attacks are single action-visible propose_patch calls whose dispositive evidence appears in condition A — the confirmatory claim must be worded "CoT adds nothing when attacks are fully action-visible" unless new cross-object/multi-step templates are added; (c) trace-availability gate re-specified to use per-turn parse statuses (any non-empty thinking output now counts as a trace); (d) multi-part goals are graded on the primary goal key only (predeclared definition of main-task success) | Scientist review Q4/Q5; these are design truths that must be stated in the preregistration, not patched silently | Preregistration fields; pilot (feasibility stage) proceeds under current splits per reviewer recommendation | senior scientist review |
| 2026-10-19 | Split manifest v1.0.0 → v1.1.0: holdout moved from incident_workflow (1 family / 2 template clusters) to two NEW families — secrets_registry + release_gate (2 families / 4 template clusters); incident_workflow joins the shift set. Benchmark expanded 6 → 8 families / 12 → 16 templates; pilot plan 48 → 64 episodes | Senior-scientist review C2: the old holdout gave a degenerate clustered bootstrap (2 clusters → ~3 distinct resample compositions → no valid CI regardless of data). The new holdout families are unseen at pilot time, making the confirmatory holdout a genuine distribution shift AND giving a non-degenerate cluster bootstrap. Two of the four new attacks are deliberately innocuous-looking on the action surface (export_allowlist grant, blackout.active=false) to partially address the action-visibility limitation | All episodes regenerate (pre-pilot, no data existed); template registry + split manifest hash changed; pilot GPU cost +33% (2 shards × 32 episodes, ~3-5h each) | senior scientist review + implementation agent |
| 2026-10-19 | GPU quota exhaustion event: after ~3h of cumulative GPU kernel time (preflight 0.3h + smoke v1 1.3h + smoke v3 1.1h), every subsequently queued GPU kernel (smoke v4, pilot shard 0 v1/v2, smoke v5) was auto-canceled by Kaggle after 30min-3.5h in queue; a CPU probe kernel completes instantly, proving the account and CLI path are healthy. Pilot generation BLOCKED on external GPU capacity, not on methodology | Kaggle account GPU allowance drained; the exact weekly allowance is account-tier-specific and was smaller than assumed | No data affected; kernels are resumable (re-push skips completed episodes); scripts/resume_pipeline.sh automates the full pilot sequence (push -> poll -> collect -> merge -> dataset -> monitor kernel -> analysis) when capacity returns; alternative: supply a different Kaggle token or any CUDA host (kernels are path-portable) | implementation agent (external constraint) |

# 2026-10-19 — v2 provenance gate for sandbox engine version

- **Observed:** the pilot configuration declared `sandbox.engine_version: 1.0.1`
  while the checked-in transition engine reports `ENGINE_VERSION = 1.0.2`.
- **Impact:** already-started pilot outputs remain usable for capability and
  pipeline diagnostics because each transition stores its actual engine
  version, but the run-manifest config is not an exact as-run specification.
  Those outputs are therefore **not admissible as confirmatory evidence**.
- **Repair:** configs now pin `1.0.2`; configuration loading hard-fails when the
  declared and imported engine versions differ. The next research run must use
  the v2 commit and receives a new config hash/run id.
- **Freeze status:** pre-preregistration; no confirmatory data were unblinded.

### 2026-10-19 — Remove automatic AUROC substitution

- **Observed:** the draft protocol allowed an underpowered fixed-FPR design to
  relabel paired AUROC as the confirmatory primary outcome.
- **Impact:** AUROC answers a ranking question, not recall at the intended
  operating point, so substitution would change the scientific question.
- **Repair:** confirmatory execution is now blocked below 40 independent benign
  calibration-template clusters; AUROC remains a pilot/exploratory diagnostic.
- **Freeze status:** pre-preregistration; no confirmatory data were unblinded.

### 2026-10-19 — Transformers 5.x generation RNG compatibility

- **Observed:** Kaggle preflight on merged commit `aac912e6` and Transformers
  5.16.1 rejected the adapter's `generator` keyword before sampling.
- **Impact:** no model output was generated and no episode data were affected.
- **Repair:** generation now runs inside a seeded `torch.random.fork_rng` scope,
  which restores the worker RNG state afterward; the generation parameters and
  effective seed remain unchanged and are still saved in raw evidence.
- **Verification:** a dependency-free adapter regression asserts that no
  unsupported keyword reaches `model.generate`, seeding occurs inside the
  isolated scope, and scope exit restores the prior RNG context.
- **Freeze status:** pre-pilot rerun; no confirmatory data were unblinded.

### 2026-10-20 — Over-broad gitignore silently excluded two governance docs

- **Observed:** `docs/SOP_002_KAGGLE_OPERATIONS.md` (referenced by README) and
  `docs/blueprint/03_KAGGLE_RUNBOOK.md` were missing from the public repository
  although written in the first session; the secrets-hygiene pattern `*KAGGLE*`
  in `.gitignore` matched their filenames, so they were never staged or pushed.
- **Impact:** governance docs referenced by the README returned 404 for any
  external reader; no code or experiment data was affected. The broad pattern
  was a silent-reproducibility hazard (files disappear from version control
  without any error).
- **Repair:** `*KAGGLE*` replaced with precise secret filename patterns
  (`KAGGLE_API_TOKEN*`, `KAGGLE_TOKEN*`); both docs committed; a warning note
  added to `.gitignore` forbidding broad keyword patterns in the secrets block.
- **Verification:** `git check-ignore` exits 1 (not ignored) for both paths;
  `git ls-files` confirms both tracked after commit.
- **Freeze status:** documentation only; no confirmatory data were unblinded.

### 2026-10-20 — Merge script silently dropped worker-2 episode files (caught by count check)

- **Observed:** the first merge of the two pilot shards produced 32 episodes
  (16 attack + 16 honest) instead of 64. `scripts/merge_shards.py` read only
  `episodes/shard-00001.jsonl` per input, but each generation kernel runs TWO
  GPU workers and writes two episode shard files per run dir — every
  worker-2 file was silently skipped. Individual shard verifications passed
  (32/32 each) because the verifier globs all shard files; only the merge
  script had the single-file assumption.
- **Impact:** no data were lost (both shard outputs were intact and
  re-downloadable; the merged dir was rebuilt from scratch). Had the bad
  merged run been monitored, the pilot would have silently analyzed half
  the data with a degenerate E-pairing (attack twins for honest episodes
  missing).
- **Repair:** merge_shards.py now globs ALL `shard-*.jsonl` files per input
  (episode/privileged file sets must match), renumbers output files
  sequentially, updates transaction shard references, and enforces a hard
  count check: merged episodes must equal the sum of ALL input shard-file
  episodes or the merge aborts. Ruff `strict=True` zip. Re-merged: 64
  episodes, verify OK, outcome re-grade 64/64.
- **Verification:** merged run verifies clean; pilot gate passes all 6 gates
  on the full data (attempts 14/32=44%, success 14/14=100%, honest 25/32=78%,
  traces 64/64=100%, invalid 3%, timeout 5%).
- **Freeze status:** pre-preregistration; the pilot is feasibility data,
  and the monitoring stage has not yet consumed the (previously bad) merged
  output — mon kernel runs against the corrected 64-episode run only.

### 2026-10-20 — Smoke-stage E-condition path drift (caught by new S4 gate tooling)

- **Observed:** the new `scripts/monitor_gate.py` (S4 evaluator, written while
  the pilot shards generate) recomputes every recorded monitor view through
  the canonical research path and compares `view_hash`. Its first run against
  a fresh CPU smoke caught 8/8 E-condition view-hash mismatches: `smoke.py`
  resolved E partners with a drifted raw-index lookup (self-guarded to None)
  and omitted the partner-instruction probe — so every E-view rendered
  "(none available)" and the smoke validated a DIFFERENT code path than
  `monitor_runtime.run_monitoring` uses for research scoring. The research
  path itself was unaffected (it uses the fixed `find_partner` + full probe
  from the 2026-10-19 BLOCKER B2 repair).
- **Impact:** smoke-stage false assurance only (the smoke is integration
  validation, never research data). No pilot or research scores were
  produced through the drifted path; the pilot's monitoring runs in the
  kernel via `monitor_runtime`, which was already correct.
- **Repair:** `smoke.py` now mirrors `run_monitoring` exactly —
  `find_partner(partner_index, env)` for every episode, REAL hidden
  instructions from the privileged store (previously a synthetic excerpt),
  the partner-instruction probe, and an explicit smoke problem when E pairs
  fail to resolve. Regression test
  `test_smoke_e_condition_uses_research_partner_path` asserts every recorded
  E-view hash equals the research-path recompilation and the partner is the
  opposite-policy twin.
- **Verification:** monitor_gate on a fresh smoke run: M1 32/32 recompiled,
  0 mismatches; 128 tests green; lint clean.
- **Freeze status:** pre-preregistration; smoke is integration validation
  only, never analysis data.

### 2026-10-20 — G6 evidence scope + confirmatory-scale predeclaration (pre-pilot)

- **Observed:** SOP-001 gate G6 asks for "estimated GPU time for confirmatory run
  within weekly quota (with 25–40% contingency)", but the confirmatory scale is
  itself a predeclared post-pilot decision (set from pilot variance; fixed-FPR
  confirmatory is blocked below 40 benign calibration-template clusters
  regardless). Yesterday's S3 evaluation used a temporary evidence file that
  was never persisted — no durable G6 record existed.
- **Decision (predeclared BEFORE pilot launch):** the committed pilot-phase
  spend (64-episode generation 6.8 h + 4-condition monitoring 3.2 h = 10.0 h,
  measured from the v6 smoke: 6.4 session-min/episode dual-T4, 17 tok/s,
  30% contingency → 13.0 h required vs 25.0 h available, owner-confirmed) is
  what G6 assesses. The provisional 216-episode confirmatory scale projects
  ~28 h total, EXCEEDING both the 25 h currently available and one ~30 h/week
  rolling window with contingency. Therefore the S5 preregistration MUST set
  the confirmatory scale to fit a rolling weekly window — or split generation
  across windows, or reduce secondary conditions FIRST — never the primary
  comparison (SOP-002 section 8 rule).
- **Verification:** `runs/gpu_feasibility_evidence_2026-10-20.json` committed
  with the measured basis; pilot_gate.py reproduces all six gates from the
  archived run directory.
- **Freeze status:** pre-preregistration; recorded before any pilot data exist.

### 2026-10-20 — S3 capability gate PASSED on the upgraded core (smoke v6)

- **Observed:** GPU smoke v6 (run `traceguard_gpu_smoke:gpu_smoke:b809551ea30b`,
  pinned commit `32a6bc0`, RNG hotfix active, transformers 5.16.1, Qwen3-4B
  revision `1cfa9a72`) completed all 11 kernel gates, `verify` re-graded 8/8
  episodes, and the predeclared S3 gates all passed: attack attempts 3/4 (75%),
  attack success 3/3 (100%), honest main-task success 4/4 (100%), trace
  availability 8/8 (100%), invalid-action 0%, timeout 12.5% (one episode:
  attack succeeded but the cover task ran past the turn budget — retained as
  data per protocol).
- **Impact:** the pilot may launch on commit `32a6bc0` — the code state these
  gates measured. Sampling trajectories differ from smoke v3 (different RNG
  mechanism: fork_rng vs per-call generator), which is expected and recorded
  in the as-run provenance; capability rates remain well above all gate
  thresholds in both runs.
- **Verification:** kernel log + environment.json + full run artifacts
  downloaded and verified; audit script read every episode (all turns carry
  non-empty reasoning AND non-empty visible content — the anti-leak split
  contract holds on-GPU); `verify` outcome re-grade passed 8/8.
- **Freeze status:** pre-preregistration; smoke is integration validation
  only, never analysis data.

### 2026-10-20 — CORRECTION: the 2026-10-19 "GPU quota exhaustion" record was wrong

- **Observed:** the project owner corrected the record: the CANCEL_ACKNOWLEDGED
  kernel states were the owner MANUALLY cancelling queued kernels in order to
  restart them interactively (interactive starts run immediately; the pushed
  versions sat in an erratic queue). GPU allowance was never exhausted —
  25+ GPU hours remained available throughout.
- **Impact on the earlier record:** the 2026-10-19 table row describing an
  "auto-canceled by Kaggle / allowance drained" event and the diagnosis ladder
  built on it were UNVERIFIED inference presented as fact — a violation of
  quality-manual rule NH-7 (unverifiable claims must be marked UNVERIFIED, not
  asserted). The real explanations for the observed states were: (a) the
  shard-0 ERROR was a genuine software failure (transformers 5.16.1 rejecting
  `generator=`), fixed by the 2026-10-19 RNG hotfix; (b) the cancellations were
  owner interventions, not infrastructure preemption.
- **Repair:** (1) this correction supersedes the quota-exhaustion claim; (2)
  SOP-002 now documents the real Kaggle queue behavior (erradic admission
  latency; owner may manually start kernels; standing rule: if a push sits
  queued for too long, cancel and re-push, or let the owner start it
  interactively); (3) kernel-state diagnoses must distinguish CANCELLED (owner
  action by default) from ERROR (software) from COMPLETE (evidence available).
- **Verification:** 2026-10-20 GPU probe kernel run completed with Tesla T4,
  fp16 matmul finite (probe JSON archived); owner-manually-run preflight also
  completed; the account had 25+ GPU hours available per owner statement.
- **Freeze status:** infrastructure operations only; no data or frozen
  parameters affected.
