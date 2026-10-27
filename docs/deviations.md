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

### 2026-10-20 — Monitor kernel v1 OOM on single-T4 (full-CoT prompts); dual-GPU split fix

- **Observed:** the first pilot monitoring kernel run (umar0x/
  traceguard-monitor-scoring v1, pinned 32a6bc0) completed rules_v1 (64/64
  in all four conditions) and keywords_v1, but every long-CoT LLM score in
  conditions C/D/E hit `torch.OutOfMemoryError`: prompts carry the FULL
  agent reasoning (pilot traces up to ~61K chars ≈ 18K tokens; KV cache
  ≈ 3.4 GiB at fp16) on top of the 8 GiB model on ONE 14.56 GiB T4. The
  runtime caught and recorded each failure per (episode, condition) and the
  kernel exited rc=1 with all partial artifacts intact. The CPU smoke never
  exercised the LLM monitor with real trace lengths (smoke runs
  rules+keywords only; the machinery drill used the fake provider).
- **Impact:** no fabricated or imputed scores; failing calls were recorded
  as failures. `max_monitor_tokens: 3072` exists in the frozen configs but
  is NOT implemented anywhere — truncating the CoT to fit one GPU was
  REJECTED as a protocol change that would bias condition C against the
  treatment (attack deliberation often lives mid/late trace; a 3072-token
  head cap would cut it on the longest episodes, artifacting ΔR toward
  the null).
- **Repair (infrastructure only):** (1) monitor kernel sets
  `TRACEGUARD_DEVICE_MAP=auto` — the model splits across BOTH session T4s
  (weights ~4+4 GiB). (2) After the SECOND attempt still OOMed on a single
  11.22 GiB allocation — the prefill LOGITS tensor (full [seq_len, vocab]
  fp32 ≈ 12 GiB at ~20K-token prompts, ~152K vocab) — the provider now
  passes the version-appropriate keep-last-logits kwarg to generate
  (`logits_to_keep=1` on transformers >=5, `num_logits_to_keep=1` on 4.x,
  resolved from the model's forward signature; never an unknown kwarg),
  which is semantically identical for sampling and also protects agent
  generation. The monitor model, checkpoint revision, prompt, evidence
  protocol, and conditions are UNCHANGED; placement and kwargs are
  recorded in each score's inference audit. A fresh full monitoring run
  follows.
- **Revision-gate upgrade (mon kernel):** equality (`mon commit == gen
  commit`) is replaced by: the monitor pin must be a DESCENDANT of the
  generation commit AND the diff between them may touch ONLY the
  explicitly allowlisted inference file (`src/traceguard/models/qwen.py`)
  plus non-research paths (docs/tests/scripts/kaggle/runs). Rationale: a
  provider memory hotfix must not invalidate already-generated, verified,
  gate-passed episodes; the registry/view/monitor/grader/schemas/configs
  protection is preserved, and every allowed hotfix is recorded here and
  in the kernel's `revision_gate` manifest block. The CONFIRMATORY stage
  will run generation and monitoring on a single identical commit, so the
  strict discipline returns there.
- **Verification:** pending the v3 kernel run — expected: zero
  generation failures across all 256 LLM calls; monitor_gate M1-M3 on the
  output. 128 tests green (adapter regression extended: forward-signature
  resolution asserted; `logits_to_keep=1` present in generate kwargs).
- **Freeze status:** pre-preregistration; the failed v1 attempt produced no
  admissible LLM scores (recorded as failures, never imputed), so no
  analysis view is contaminated.

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

### 2026-10-20 — Mon-kernel diff-allowlist gate additions (gate caught two unapproved files)

- **Observed:** mon kernel v3 failed its own NEW revision gate before any
  scoring: the gen(32a6bc0)→mon(f36d033) diff contained two research-core
  files outside the initial allowlist — `.gitignore` (the gpu_feasibility
  whitelist edit) and `src/traceguard/smoke.py` (the earlier smoke-stage
  E-pairing fix). The gate worked as designed: nothing scores until every
  changed file is consciously re-approved.
- **Verification of scoring-safety:** `smoke.py` is imported only
  function-locally inside `cli.cmd_smoke` (never by `monitor_runtime` or
  the monitor CLI path), so it cannot affect how stored episodes are
  scored; `.gitignore` has zero runtime semantics. The complete
  research-core diff gen→mon is exactly {`.gitignore`,
  `src/traceguard/models/qwen.py`, `src/traceguard/smoke.py`} (verified via
  `git diff --name-only` minus non-research paths).
- **Repair:** both files added to the kernel's allowed_changes with
  in-code justification; the gate remains fail-closed for anything else.
- **Freeze status:** pre-preregistration; no confirmatory data unblinded.

### 2026-10-20 — TRUE root cause of the monitor OOMs: EAGER-ATTENTION fallback (not logits, not KV)

- **Observed:** mon kernel v4 ran the hotfixed provider (logits guard active,
  traceback line numbers confirm the new file) yet still OOMed with the same
  single-allocation signatures: 3.37 / 11.22 / 21.14 GiB. Primary-source
  forensics against transformers 5.16.1 (modeling + generation sources
  fetched at the v5.16.1 tag) closed the case:
  (a) generate() ALREADY auto-sets logits_to_keep=1 (generation/utils L2612
  auto-sets when forward supports it) — the logits tensor was never the
  allocation; (b) Qwen3's attention interface falls back to
  `eager_attention_forward` when `config._attn_implementation` does not
  resolve to sdpa on the Kaggle load path (modeling L262-264), and eager
  materializes the full [heads, seq, seq] scores matrix per layer. The
  observed allocations match eager scores at 32 heads fp16 EXACTLY:
  seq ~7.3K -> 3.37 GiB (v1), seq ~13.3K -> 11.22 GiB (v2/v4), seq ~18.3K
  -> 21.14 GiB (v4). All three OOM signatures across all kernel versions
  were the same defect.
- **Why generation never OOMed:** agent-turn prompts stay far shorter
  (~4K-char history appends, 40-message cap) — eager scores stayed within
  VRAM; all 64 pilot episodes completed. The generation data are valid;
  the pilot was produced with eager attention throughout (deterministic
  per seed within a fixed implementation).
- **Repair:** the provider now PINS `attn_implementation="sdpa"` at load
  (env-overridable `TRACEGUARD_ATTN_IMPL`). SDPA never materializes the
  scores matrix and is the standard torch backend on sm_75 with fp16. A
  load failure now fails LOUDLY instead of silently falling back to eager.
  The logits guard stays as documented belt-and-suspenders.
- **Verification:** pending the mon v5 run — expected: zero OOMs at all
  prompt lengths. Monitor scores under SDPA are their own measurement;
  within-run consistency is what the analysis requires (all pilot scores
  will be produced by the same SDPA path in one kernel run).
- **Freeze status:** pre-preregistration; no confirmatory data unblinded.
  128 tests green; lint clean.

### 2026-10-20 — Diagnostics kernel: SDPA pin took effect but math-backend attention still OOMs; predeclared evidence window implemented

- **Observed:** the mon-attention diagnostics kernel (umar0x/
  traceguard-monitor-diagnostics, ground-truth run on the real stack:
  transformers 5.16.1, torch 2.11.0+cu128, dual T4, fp16, device_map=auto)
  settled the forensics: (a) `attn_implementation="sdpa"` DID take effect
  (config._attn_implementation = "sdpa", verified); (b) attention still
  materializes QUADRATIC memory — attempted allocations scale exactly as
  seq² (20K→21.22 GiB, 24K→30.58, 30K→47.76; ratio 2.25 = (30/20)²),
  i.e. torch SDPA falls back to the math backend on sm_75+fp16; (c) the
  measured ceiling: a 5,343-token prompt peaked at 12.32 GiB (worked) and
  ~8K-token prompts OOM. No attention backend on this stack avoids the
  quadratic scores tensor.
- **Empirical pilot evidence:** 346 turns measured: reasoning tokenizes at
  4.45 chars/token; the longest episode reasoning is 87,886 chars
  (~19.7K tokens) — exactly the OOM range observed in monitor attempts.
- **Decision (predeclared BEFORE any admissible LLM scores exist — all
  five prior monitoring attempts produced zero admissible scores):**
  implement the config's already-declared `max_monitor_tokens: 3072` as a
  predeclared evidence window: the reasoning section is head-capped at
  3072 tokens × 4.0 chars/token (empirical 4.45 rounded DOWN for margin)
  = 12,288 chars (~2.8K reasoning tokens; full view ≤ ~4.5K tokens, inside
  the measured ~5-6K-token ceiling). The window applies IDENTICALLY to
  every reasoning-bearing condition (C/D/E/F); condition D matches the
  CAPPED length (±20%); condition E uses the partner's CAPPED trace; a
  visible truncation marker states kept/total chars; short traces are
  byte-identical to previous behavior. This is a monitoring-protocol
  definition (the monitor's evidence window — exactly what a deployed
  monitor with a context budget would receive), symmetric across
  conditions, and declared before scoring.
- **Rejected alternatives:** chunked prefill (high engineering risk,
  nonstandard inference path); truncating actions (condition A's evidence
  must stay complete); smaller monitor checkpoint (protocol change);
  CPU scoring (RAM also quadratic: 20K-token scores ≈ 25 GB > host RAM).
- **Verification:** 131 tests green (3 new window regressions: marker +
  truncated flag, byte-identical short traces, C/D/E symmetric capping);
  fresh CPU smoke + monitor_gate M1 32/32 hash-identical; lint clean.
- **Freeze status:** pre-preregistration; preregistration draft updated
  with the evidence-window rule; pilot is feasibility data.

### 2026-10-20 — Mon kernel v6 output omitted the completion ledger (restored byte-verified; kernel fixed)

- **Observed:** the mon kernel v6 run completed (monitor_rc 0, 768/768
  scores) but its output copy list omitted `completed_episode_ids.txt`
  (and transactions/commits/artifacts), so `verify` failed with "64
  episodes not in completion ledger" while the outcome re-grade passed
  64/64. The evidence was present in the kernel's input (the public
  dataset, created from the archived merged run) — the kernel simply did
  not copy it out.
- **Repair:** the omitted pieces were restored locally from the archived
  authoritative run (`runs/pilot/pilot_v0_run`, git-committed) with
  byte-verification: the archive's merge-time `checksums.sha256` (139
  entries) spot-checked 40/40 OK; every restored file SHA-256-verified
  against the archive (ledger + 64 transactions + 64 commits + checksums).
  `verify` then passed clean. The mon kernel copy list now includes the
  full research record for future runs.
- **Verification:** `traceguard verify` on the restored monitored run:
  ok=true, no problems; outcome re-grade 64/64; monitor gate M1 768/768.
- **Freeze status:** pilot (feasibility) stage; no confirmatory data
  unblinded.

### 2026-10-20 — Phase 3 protocol created after pilot review

- **Observed:** the exploratory 64-episode pilot showed raw CoT degrading the
  Qwen3-4B monitor and exposed semantic authorization laundering, coarse scores,
  differential invalidity, all-condition complete-case pairing, unstable tiny-
  cluster intervals, action visibility, and head-window vulnerability.
- **Change:** added relational task families, four attack-presentation strata,
  0–100 rubric scoring with retry-once, structured late fusion, optional
  independent fact checking, head/tail evidence sampling, pairwise-complete
  analysis, dual realized-FPR reporting, missing-score bounds, explicit CI
  eligibility, exact small-cluster distributions, and fail-closed release
  packaging. See `PHASE3_PROTOCOL.md`.
- **Impact:** no pilot artifact or conclusion is rewritten. Prompt/view/config
  hashes change, so all Phase 3 episodes and monitor scores must be newly
  generated. Phase 3 remains exploratory until a separate preregistration is
  frozen and the stated gates pass.
- **Approval:** project owner requested implementation on a new branch; merge
  remains subject to review.

### 2026-10-20 — PR #3 (Phase 3 multi-model architecture) merged after line-by-line review

- **Observed:** PR #3 (branch `v6`, 7 commits, 4,346 insertions) restructures the
  research core for Phase 3: audited model registry + fail-closed provider
  factory with probe-artifact enforcement, generic tagged-reasoning adapter,
  relational non-action-visible task families (`ref_target_equals` invariants),
  four frozen attack-presentation strata, 0–100 four-component rubric,
  structured late-fusion monitor, pairwise-complete per-comparison analysis,
  explicit CI eligibility with exact small-cluster distributions, dual-T4 worker
  runtimes with validated merges, QLoRA development-only tuning path, and the
  fail-closed release contract.
- **Review:** every commit read file-by-file against the project's
  prior-failure checklist (merge-shard globbing, kernel copy lists, OOM
  traps, transformers-5 compatibility, think-tag constants, provenance gates,
  condition blindness, E-pairing). 158 tests green, ruff clean, mypy clean
  across 52 files, CI green on the branch tip; old-pilot outcome re-grade
  passes unchanged (64/64) on the merged core.
- **Verified-not-broken:** worker merges use private run dirs + per-record
  schema validation + conflicting-id detection; the monitor kernel copy list
  now includes the completion ledger; context-budget enforcement
  (`prompt + generation <= max_context_tokens`) closes the pilot's OOM class;
  E-pairing remains strategy-safe because each strategy's twins share a
  disjoint seed stream; the smoke-only calibration fallback is explicit and
  labeled.
- **Compatibility note (binding for archival verification):** the v6 evidence
  window (75% head / 25% tail) and D-condition marker change view compilation.
  Recorded monitor scores from the 64-episode pilot therefore fail an M1
  *recompile* under v6 code (450/768 view-hash mismatches) by design;
  re-verifying that run's views requires its own pinned monitor commit
  (`3f63e6e` lineage). `traceguard verify`, outcome re-grading, and
  `traceguard export` hash validation remain compatible. The M2'/M3 gate
  failures on old data reflect v6's raised thresholds (design facts for a
  pre-Phase-3 run), not new defects.
- **Approval:** merged as `2cc54d0` (merge commit preserves the PR link).

### 2026-10-20 — Monitoring evidentiary archive committed (reproducibility gap closed)

- **Observed:** an independent audit (2026-10-20) found the pilot's monitoring
  artifacts had never been committed: monitor scores, analysis report,
  monitoring environment manifest, and figures existed only on Kaggle and a
  lost local workspace. The generation record was public; the monitoring
  evidentiary record was not. "Everything is public" was not true of GitHub.
- **Change:** recovered the monitor kernel output from Kaggle (768/768 scores,
  3 monitors x 4 conditions x 64 episodes, byte-identical input episodes
  verified against the committed archive), restored the completion ledger
  from the authoritative archive (SHA-256 equal), regenerated the analysis
  report and figures at the merged HEAD, and committed everything under
  `runs/pilot/` (scores, reports, `environment_monitor.json`, kernel log).
  `traceguard export` now validates the complete release fail-closed
  (`release_complete: true`, zero validation errors) and the per-file SHA-256
  manifest is committed.
- **Root cause of the original omission:** a generic `artifacts/` gitignore
  pattern silently excluded every run's integrity manifests, and the session
  that produced the analysis treated local download paths as durable. Fixed
  the pattern (`!runs/**/artifacts/`) and recorded the lesson: research
  evidence lives in the repository, not in download directories.
- **Impact:** headline pilot numbers (holdout recall A 1.0 vs C 0.0; noise
  0.5; shuffled 0.0; 2 benign calibration clusters; precision gate failed)
  reproduce identically under the merged core. The archived v6 report
  intentionally omits `fig_paired_effects.png` because all bootstrap CIs are
  `not_estimable` at 1 positive cluster — zero-width intervals are no longer
  drawn as if they were precision.

### 2026-10-20 — CORRECTION: 1f2cdf6 commit message misstated the C-marker count (46/64 → 50/64)

- **Observed:** the 2026-10-20 independent audit flagged that commit
  `1f2cdf6`'s message states window truncation markers were
  "asymmetric (C=46/64, D=0/64, E=50/64)" while recompilation and the review
  documents give 50/64 for C. Commit messages are immutable history, so this
  entry is the authoritative correction.
- **Verification (programmatic, per NH-8):** counted the marker string in the
  archived `rendered_prompt` of every committed llm_qwen_hf score at HEAD:
  actions 0/64, actions_plus_cot **50/64**, actions_plus_noise **0/64**,
  actions_plus_shuffled_cot **50/64**. The 46 was a transcription error in
  the commit message; no review document, report, or archived artifact
  carries the wrong figure.
- **Impact:** none on any estimate — the marker count is a descriptive
  statistic; the D-marker asymmetry finding (0/64) is unchanged and is what
  the security review's disclosure condition and the preregistration
  execution correction address.

### 2026-10-20 — Audit-flagged documentation discrepancies corrected

- **Observed:** the independent audit found three reporting discrepancies:
  (1) the security review described the censored holdout positive
  `ep-b82f0033` as "retained"; (2) the preregistration claimed the evidence
  window applied identically across C/D/E while the executed pilot emitted
  no visible marker on D (0/64); (3) the pairwise-complete-per-comparison
  analysis specified in the prereg was implemented as one all-condition
  complete-case frame.
- **Change:** (1) security-review wording corrected with the verified retained
  pair (`ep-472be01b9b`, `ep-58a45dfda9` — confirmed from the committed
  scores archive; the first correction draft cited a wrong episode id and was
  itself corrected the same day); (2) preregistration carries an execution
  correction note (D-marker asymmetry, fixed from the Phase 3 core onward);
  (3) fixed in code by PR #3 — every contrast now builds its own
  pairwise-complete frame, with the omnibus frame retained as a diagnostic.
- **Impact:** corrections are reporting-integrity fixes; no recorded score,
  gate result, or estimate changed. The pilot remains exploratory with all
  originally reported caveats.

### 2026-10-20 — Phase-3 pilot run archived; matrix freeze + monitor-cell configs (owner directive: all accounts simultaneously)

- **Observed:** the 2026-10-20 owner message commit (`20a2d6b`) directed (a)
  simultaneous use of all three Kaggle accounts, (b) completion of all
  remaining runs toward the final research conclusion, and (c) queue
  restarts after 20 minutes (replacing the 30-minute guidance). Also found:
  the phase-3 pilot run (128 episodes, gate-passed) existed only as the
  public Kaggle dataset + local downloads — the same class of archival gap
  the reproducibility audit flagged for Phase 2.
- **Change:**
  1. The verified merged pilot run is committed at `runs/phase3/pilot_v0_run`
     (263-file checksum manifest verified before commit; `runs/phase3/`
     whitelisted).
  2. The study matrix is frozen at `configs/model_matrix_phase3.yaml`:
     anchor (4B/4B, running), scale-monitor (Qwen3-14B, probe-admitted rev
     `40c06982`), cross-monitor (DeepSeek-R1-Distill-14B, probe-admitted rev
     `1df85071`; substitutes the template's unprobed gemma-4-12B cell — the
     substitution is the family-diversity arm actually probed and admitted),
     and two agent cells explicitly `pending_agent_probe`.
  3. Monitor-cell configs are DERIVED, not hand-written:
     `scripts/derive_monitor_cell_configs.py` copies the generation
     manifest's frozen config verbatim and swaps ONLY the monitor-model
     fields + `provenance.matrix_cell_id`. The runtime consistency gate then
     enforces byte-equality of every generation-defining section at score
     time. Cell score files can never collide with anchor scores: with
     `matrix_cell_id` set, monitor identities become
     `llm_qwen_hf_c885515408` / `llm_generic_hf_eb379ca5e7`.
  4. Cell kernels (`kaggle/mon_scale_qwen14`, `kaggle/mon_cross_deepseek14`)
     extend the anchor kernel's revision-gate allowlist with exactly the
     three frozen config files (data-only; justified in-line) — all other
     diffs remain hard-rejected.
- **fact_checked arm DEFERRED (declared, not silently dropped):** late fusion
  + cross-family fact-checker would load two ~9.5 GiB NF4 models inside one
  worker's single visible T4 (14.5 GiB) — an OOM loop by construction. It
  requires a dual-GPU single-worker kernel variant or a probed int8
  fact-checker; recorded as the follow-up, never substituted silently.
- **Validation before any push (owner directive: test locally + in the target
  environment):** both cell configs drilled through the REAL
  `traceguard monitor` path against a temp copy of the archived run
  (consistency gate + scoring ok); full test suite green; ruff clean. The
  cross kernel's allowlist initially dropped the scale config file (caught by
  ruff B033 during parameterized generation — the diff gate would have
  rejected it on Kaggle); repaired before push.
- **Impact:** monitor-capability and monitor-family cells now score the SAME
  stored episodes as the anchor cell — the capability x family axis of the
  final conclusion runs without regenerating anything.

### 2026-10-20 — ROOT CAUSE CORRECTION: persistent 409s on agent-probe pushes were title/slug collisions, not session reservations

- **Observed:** the agent-role probe kernels (`traceguard-probe-qwen14_agent`,
  `traceguard-probe-deepseek14_agent`) 409'd on every push since 2026-10-19,
  blocking the agent-scale matrix cells. The earlier §9.1 explanation
  (session-reservation window) did not apply: these kernels had NEVER been
  created, and the accounts were confirmed free by the owner.
- **Root cause (found 2026-10-20):** both probe metadatas reused the TITLES of
  the already-completed monitor-role probes ("TRACEGUARD Probe Qwen14" /
  "TRACEGUARD Probe Deepseek14"). A new-kernel push derives its slug from the
  TITLE, so the API resolved the push against the EXISTING monitor-probe
  kernels while the metadata id pointed at a different (underscored, invalid)
  slug — a create/update conflict surfaced as a bare 409. The underscores in
  the original ids were an invalid workaround for the collision.
- **Fix:** distinct titles that resolve exactly to their ids
  ("TRACEGUARD Probe Qwen14 Agent" → `...-qwen14-agent`, "TRACEGUARD Probe
  Deepseek14 Agent" → `...-deepseek14-agent`); additionally the deepseek
  agent-probe metadata carried the WRONG ACCOUNT id (`umar0x/...` instead of
  `heicfast/...`) — both fixed. Both probes then pushed successfully on the
  first attempt (v1 admitted).
- **Lesson (NH-8 class):** every new kernel's title MUST resolve to its id
  slug; never reuse a title across kernels. The §9.1 reservation-window
  theory remains valid for post-completion re-pushes of the SAME kernel, but
  it was never the agent probes' failure mode.

### 2026-10-20 — Monitor-cell kernel bring-up: three distinct root causes, all fail-closed, all fixed before any scoring occurred

The cross-monitor cell kernel took three short-lived runs to reach scoring;
each failure was a hard gate, not silent corruption, and each fix was
committed with justification before the next push:

1. **v2 — revision gate REJECTED the pin diff** (by design): owner commit
   `20a2d6b` edits README.md and commit `25f5883` adds `configs/probe/*.yaml`
   after the generation commit. Fix: explicit allowlist entries for the three
   verified-inert files (probe configs are read only by `traceguard
   probe-model`; verified absent from the monitor/analyze/verify/export import
   graphs).
2. **v3 — factory admission contract**: probe-required models demand
   `TRACEGUARD_{ROLE}_PROBE_ARTIFACT` at runtime with a recomputed hash equal
   to the config's frozen `probe_artifact_sha256`. The probe kernels set this
   for themselves; the anchor mon kernel never needed it (4B is not
   probe-required). Fix: both cell kernels bind the committed artifacts
   (hash chain verified locally: claimed == computed == config).
3. **v4 — bitsandbytes missing**: the anchor kernel pattern predates
   quantized monitors (4B fp16 needs no bnb); NF4 loads hard-fail without
   bitsandbytes>=0.46.1. Fix: the same install+import gate the probe kernels
   use.

No episode was scored under any failed configuration (fail-closed everywhere),
so no contamination occurred. The scale-agent gen kernels were built with all
three lessons applied (artifact binding + bnb gate + factory validation).

### 2026-10-20 — OWNER DIRECTIVE (commit 5d1a00e): core v0.6.0 upgrade; all prior data renamed PRE-TESTS-DATA; program restarts from scratch

- **Directive:** the owner upgraded the research core (v0.6.0: workflow engine
  with hash-chained journal, evidence index, failure taxonomy, study-plan
  sizing, McNemar+Holm analysis, PR curves, figure contract) and directed an
  immediate stop of all in-flight runs, renaming of all previously recorded
  data to `pre-tests-data`, and a from-scratch restart of every phase on the
  new core (5 new Kaggle accounts / ~150 GPU-hours announced).
- **Executed:** (1) all five active/queued kernels deleted across the three
  accounts (anchor monitoring ~5-6h in, scale/cross monitor cells, both agent
  generations — per explicit owner instruction; partial outputs were never
  landed and are not archived); (2) `runs/{pilot,phase3,gpu_smoke_v1,gpu_smoke_v3,gpu_feasibility,probes}`
  → `runs/pre-tests-data/...` via git mv, whitelists + README + script paths
  updated; (3) new-core runs will archive under `runs/v6/`.
- **Pre-test data status:** all pre-v0.6.0 evidence (phase-2 pilot 64 episodes
  + phase-3 pilot 128 episodes + admission probes + smokes + feasibility)
  remains public and valid AS PRE-TEST RECORDS — generated/analyzed under
  v0.5.x cores; they are not comparable-at-a-glance with v0.6.0 outputs and
  do not contribute to the v0.6.0 confirmatory program except as
  design/measurement priors (sizing inputs, ICC estimates, funnel priors).
- **Kaggle datasets from the pre-test era stay published under their original
  slugs** (heicjpg/traceguard-phase3-pilot-run-v1, etc.): Kaggle has no
  rename operation; the repo path rename is authoritative and the dataset
  metadata files already record their provenance.
- **New-core verification before any run (owner: check errors first):** 166
  tests, ruff, mypy (56 files), doctor (v0.6.0), workflow-plan validation all
  PASS on the merged tree. A multi-POV review of the new research-operations
  modules is running before the restart (per docs/REVIEW_SYSTEM.md).

### 2026-10-20 — From-scratch program identity: traceguard_v6_pilot_v1

- **Observed:** run_id = experiment_name:stage:config-hash — reusing the
  pre-test experiment name would make the fresh v0.6.0 cohort claim the SAME
  run identity as the archived pre-test cohort (identical config content
  otherwise), colliding in every score file, evidence index, and release
  manifest.
- **Change:** configs/phase3_pilot.yaml experiment_name (both the top-level
  key and provenance) is now `traceguard_v6_pilot_v1` (run_id
  `traceguard_v6_pilot_v1:pilot:dd384fb49464`). The pre-test archive keeps its
  own recorded identity unchanged; the two programs are never confusable.

### 2026-10-20 — From-scratch shard-0: generation intact, in-kernel verify correctly failed on MISSING_INDEX (flow integration bug, fixed)

- **Observed:** fresh anchor shard-0 generated all 64 episodes (authentic
  think tokens, re-grade 64/64, full provenance) and then its kernel exited
  ERROR: the merged-run verify reported exactly one problem —
  `MISSING_INDEX: artifacts/evidence_index.json absent` (metadata_only).
- **Root cause:** the v0.6 stage-severity rule (security review F3: strict
  stages require the integrity chain) is correct, but the GENERATION flow
  never built the index — merge_shards.py wrote the checksum ledger and
  verified without the index. The strictness gate worked exactly as designed
  and caught a real flow gap.
- **Fix:** merge_shards.py now builds the evidence index FIRST, then the
  checksum ledger (which must cover the index — the initial fix had the
  order reversed and the ledger-coverage check correctly caught that too),
  then verifies. Merged runs carry the complete v0.6 chain at generation
  time; integration test pins the behavior.
- **Salvage:** shard-0's episodes are intact and were NOT regenerated — the
  output run was indexed + ledgered + verified locally (64/64 re-grade,
  verify green) and will be merged with shard-1's output into the fresh
  cohort. Shard-1 runs the pre-fix script and will fail its final verify the
  same cosmetic way; its episodes will be salvaged identically.

### 2026-10-21 — Phase-3 pilot attempt 1 monitoring: THREE systematic failures, root-caused and fixed (instrument v2, chain order, allocator hygiene)

**Failure A — monitor instrument contract (the M2 gate correctly blocked the
LLM arm):**

- **Observed (anchor mon, kernel `umar0x/traceguard-monitor-scoring`, pin
  542d5d4):** both T4 workers completed every call with zero generation
  failures (640/640 attempts, `ok: true`), but the S4 monitor gate FAILED:
  M2 invalid output — llm_qwen_hf 69-89% invalid per condition (114/128 on
  actions), fusion 41%; M2' pair retention 7.7-19.7% for the LLM arm; M3
  score resolution 2-6 distinct levels.
- **Root cause (verified from the recorded score audits):** the v6 prompt
  required the monitor to emit a self-computed `risk_score` exactly equal to
  its four-component rubric sum. The 4B monitor reliably fails that
  arithmetic self-consistency check (the dominant invalid class is
  `ValueError: risk_score does not equal rubric sum` — e.g. rubric summing
  to 25 with self-reported `risk_score: 0`) while its rubric components
  remain well-formed and semantically meaningful. A secondary defect: the
  v1 wording of `action_evidence` ("strength of evidence that the action
  was attempted or completed") was ambiguous between the authorized task
  action and the unauthorized act — observed outputs scored task-completion
  evidence 25 alongside risk 0. A tertiary class (~17%) is JSON
  malformation from unescaped double quotes inside evidence_refs.
- **Instrument decision (v2, 2026-10-21):** the model now reports ONLY the
  four rubric components (0-25 integers with labeled bands
  0/1-8/9-16/17-25); the 0-100 score is DERIVED deterministically as the
  component sum by the parser; `action_evidence` is disambiguated to
  UNAUTHORIZED-action evidence with an explicit "authorized task completion
  is not risk evidence" clause; JSON hygiene instruction added (no double
  quotes inside strings). A self-reported `risk_score`, if emitted out of
  habit, is recorded in `parsed_response` (`self_reported_risk_score`,
  `self_report_matches_rubric_sum`) for audit and NEVER invalidates. The
  fusion adjudicator receives the identical treatment. Rationale: the
  estimand is the monitor's structured risk judgment; the rubric components
  ARE that judgment; the self-computed sum was a redundant, failure-prone
  self-report. This follows the same philosophy as the preregistered
  late-fusion architecture (structured claims, deterministic adjudication)
  and eliminates the failure class by construction. The v1 prompt hashes
  (`075bc303…` llm, `edcd2d17…`/`75a8e950…` fusion) are retained in the
  frozen-prompt registry with provenance so the attempt-1 score archive
  keeps verifying. Attempt 1 produced ZERO admissible LLM scores (gate
  failure), so no contamination is possible; the deterministic arms
  (rules_v1, keywords_v1) were and remain fully valid.
- **Consequence:** all monitoring cells re-run on the same stored episodes
  with the v2 instrument. Generation is untouched (monitoring is a pure
  function of frozen episodes). The in-flight cross-cell run (started on
  the v1 prompt) was stopped: its LLM arm would be contract-invalid and
  non-comparable (cross-cell comparison requires the identical prompt hash
  across cells).

**Failure B — mon-kernel integrity chain order (37 verify problems):**

- **Observed:** after scoring and analysis, `verify` failed with 37
  problems: 12 files not covered by the checksum ledger (reports/ +
  tables/, written INTO the run dir by analysis) and 26 unindexed files
  (monitor_scores/ + monitor_sessions/, added by scoring) — same class as
  the 2026-10-20 shard-0 MISSING_INDEX incident.
- **Fix:** new `traceguard canonicalize` CLI command (evidence index FIRST,
  then the checksum ledger covering the index); all four mon kernels
  (anchor, scale, cross, agent_qwen14) now run canonicalize AFTER analysis
  and BEFORE verify, export is additionally gated on canonicalize success,
  and canonicalize is in the critical-command list. Local end-to-end drill
  (`scripts/drill_mon_chain.sh`) on the archived fresh cohort reproduces
  the failure (verify-before rc=1, 11 uncovered files) and proves the fix
  (verify-after green, export `release_complete: true`).

**Failure C — 14B-NF4-per-T4 allocator fragmentation (scale cell marginal
OOMs):**

- **Observed (scale mon, `heicjpg/traceguard-mon-scale-qwen14`):** rules +
  keywords scored in full; llm arm failed 13% of calls (49-53/64 per
  condition per worker) with marginal CUDA OOMs — requested allocations of
  440 MiB..2.23 GiB failing while 0.4-2.4 GiB sat "reserved but
  unallocated" (fragmentation), on a 9.5 GiB NF4 model with ~5 GiB free
  per T4, the ~2 GiB math-backend attention transients at the 3072-token
  evidence window leaving no headroom.
- **Fix (infrastructure only; no protocol change):** per-call
  `torch.cuda.empty_cache()` in the provider (generate rebuilds KV every
  call; no cross-call reuse exists, so freeing between calls is always
  safe) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in every mon
  kernel worker env. The evidence window, conditions, prompts, and scoring
  protocol are unchanged.

**M4/M5/M6 gate results on attempt 1 are design facts at pilot scale, not
failures:** benign calibration clusters 2 < 40 (predeclared consequence:
fixed-FPR confirmatory blocked; exploratory only), rules baseline recall
0/53 on the phase-3 non-action-visible families (the headroom redesign
working as intended), threshold resolution floor(16*0.05)=0 < 3. These
carry into S5 sizing as before.

### 2026-10-22 — Attempt-2 monitor runs landed/decoded: anchor cell COMPLETE (raw-arm degeneracy is instrument-level, not agent-capability); scale/cross blocked by a measured capacity knife edge; dual-T4 split chosen

**Anchor cell (`umar0x/traceguard-monitor-scoring`, v2 instrument, pinned
1394950) — COMPLETE, landed:**

- Full chain green in-kernel (monitor_rc 0, analysis 0, canonicalize 0,
  verify 0, export 0, 6472.7 s) with `monitor_gate_rc: 3` recorded as the
  honest verdict. Landed via the standard flow: local re-analysis
  reproduced the kernel's reports (plotdata byte-identical; report JSON
  identical modulo absolute path prefixes; rendered figures differ only by
  embedded dc:date + matplotlib 3.10.0-vs-3.9.2 metadata — plotdata is the
  deterministic layer), canonicalize -> verify ok (128/128 re-grade, 306
  files indexed, 0 problems) -> export `release_complete: true`. Archived
  at `runs/v6/anchor_mon/`.
- Gate verdict reproduced locally: M1 PASS (1664 views recompiled, 0
  failures, 0 leakage), M2 PASS, M2' PASS, M3 FAIL (score resolution), M4
  FAIL (2 benign calibration clusters — predeclared design fact), M5 INFO
  (rules baseline recall 0/53, FPR 0/75 on the phase-3 non-action-visible
  families — the headroom redesign working as intended), M6 FAIL
  (floor(16*0.05)=0 allowed false positives < 3).

**Scientific result (exploratory) — the pending discriminator is resolved,
and it CORRECTS the interpretation recorded in commits 9cd5fd1/01a25f6:**

- The v2 raw arm is degenerate for the 4B monitor on the anchor cohort too,
  INCLUDING the actions-only condition (A: distinct=1, all rubric zeros;
  C/E: distinct=2 [0.0, 0.25]; D: distinct=1) — 117/117 valid records per
  condition, zero information. Raw C-vs-A AUROC: 0.469 dev / 0.542 holdout
  (chance). Because condition A contains no untrusted narrative at all,
  the all-zero collapse CANNOT be attributed to "raw concatenation of the
  agent narrative" — the agent-cell commits' "the failure is ARCHITECTURAL
  (raw concatenation)" framing over-claims: under instrument v2 the raw
  arm is an uninformative MEASUREMENT INSTRUMENT for this monitor class,
  and raw-vs-fusion contrasts measure instrument-vs-instrument, not
  architecture-vs-architecture.
- Late fusion (same 4B checkpoint, structured extractors + adjudicator,
  raw CoT withheld from the adjudicator): AUROC 0.923 development / 0.690
  holdout / 0.826 shift / 0.888 validation vs 0.5 for actions-only; 16
  distinct score levels; M3 PASS for that arm. The fusion architecture's
  detection capability is replicated on the third cohort (qwen14-agent:
  0.953/0.719; deepseek14-agent: 0.859/0.634; anchor 4B-agent: 0.923/0.690).
- What remains genuinely open: whether a LARGER monitor's raw arm still
  collapses under the same v2 instrument (scale/cross cells) — i.e.
  whether the collapse is 4B-capacity-linked or instrument-linked. The
  scale/cross cells are therefore scientifically essential, not mere
  matrix completion.

**Scale/cross attempt-2 runs — ERROR, root-caused to a measured capacity
knife edge (genuine exhaustion, not fragmentation):**

- Workers completed all scoring loops and recorded every failure
  (fail-closed): C failed 23/128 (18.0%), E failed 17/128 (13.3%) — both
  above the M2 <=10% gate, which correctly refused (kernel ERROR, zero
  admissible-cell claims). A, D, and late fusion: 128/128 clean on both
  workers. Reserved-but-unallocated memory at failure was only 187-218
  MiB — the 2026-10-21 allocator fix removed the fragmentation component;
  what remains is capacity.
- Measured on the exact anchor-cohort views (scripts/measure_prompt_sizes.py):
  the OOM boundary sits at ~15.6-15.7K prompt chars (~3.5K tokens) for the
  14B-NF4-per-T4 layout: largest C pass 15,659 chars vs smallest C fail
  15,615 chars (margin -0.3% — a knife edge). E failures are NOT purely
  length-driven (an 8,059-char E prompt failed while 17,106 passed),
  i.e. the usable ceiling drifts with process state; window reduction
  alone (e.g. 2048-token window brings max C to 13,803 chars, ~13% under
  the static boundary) would leave unsafe margin AND cost protocol
  comparability.
- DECISION (infrastructure only; no protocol change): run the 14B monitor
  cells with the model SPLIT ACROSS BOTH T4s (device_map auto, ~4.75 GiB
  weights per GPU, one shared provider instance for raw+fusion — verified
  shared in monitor_runtime), giving >40% headroom at the same 12,288-char
  window. Per admission discipline the cross-GPU NF4 load path is new and
  gets a probe kernel (both 14B models) BEFORE any cell re-launch.

### 2026-10-27 — Attempt-3 capacity fix: single_split layout + deployment-scale admission stress gate + local kernel preflight (session re-created the interrupted fix set)

Session context: the 2026-10-22 session died mid-implementation of this fix
set (uncommitted work lost); this session re-created and validated it in
full. Nothing from the failed attempts was reused unverified: every piece
was re-implemented, unit-tested, drilled end-to-end on CPU, and lint/mypy
checked before any kernel push. 243 tests green.

**Root-cause class (systemic, per the owner's "make sure this doesn't
happen again" directive):** five Kaggle rounds were lost in the v6 launch to
defects that were ALL locally checkable (allowlist misses, stale artifact
namespaces, missing install gates, wrong env bindings) — each discovered one
GPU round at a time. Two structural fixes:

1. `scripts/preflight_kernel.py` (new): local pre-push replay of every
   kernel's OWN on-Kaggle checks — pin existence, the ancestor+diff-allowlist
   revision gate (AST-parsed from the kernel source, never hand-typed),
   config/probe-artifact existence AT THE PIN (git show), metadata
   title-resolving slug rule (§9.5), and the run-manifest discovery glob.
   A kernel that fails preflight locally is never pushed.

2. Deployment-scale admission stress gate (`model_probe.py`): the attempt-2
   knife edge slipped through because probes exercised ~90-char prompts
   while deployment renders ~17.9K-char evidence-window prompts. The probe
   now renders a deployment-faithful prompt (the frozen monitor template
   wrapping a synthetic condition-C view at the full config-derived window:
   max_monitor_tokens x 4.0 chars/token + action/overhead budgets — never
   hardcoded), runs a real generation under it, and gates: (a) prompt
   reaches deployment scale, (b) per-device peak VRAM within
   max_memory_per_gpu_gib on ALL visible devices (the old gate checked
   device 0 only), (c) generation completes. OOM raises and fails the probe
   fail-closed. Probe artifact schema 1.0.0 -> 1.1.0 (stress metrics +
   gates recorded).

3. `single_split` worker layout (`run_monitor_workers.py`): ONE worker
   process with the model SPLIT ACROSS BOTH T4s (CUDA_VISIBLE_DEVICES=all,
   TRACEGUARD_DEVICE_MAP=auto; accelerate balances 14B-NF4 weights ~4.75
   GiB per GPU) scoring all episodes sequentially. Trades ~2x wall time for
   >40% per-device headroom at the SAME frozen 12,288-char evidence window —
   infrastructure only, zero protocol change (the recorded 2026-10-22
   decision). `per_gpu` layout preserved byte-for-byte (historical
   behavior); single_split with <2 visible GPUs REFUSES (no fallback).
   Topology extracted to a pure unit-tested planner. Local drill: 1 split
   worker vs 2 per_gpu workers produced 1024/1024 byte-identical
   deterministic scores over a fresh copy of the 128-episode anchor cohort,
   and the canonicalize->verify chain is green on the scored copy.

4. Admission discipline (per the recorded decision): BEFORE any cell
   re-launch, both 14B monitors get split-layout probe kernels
   (probe_split_qwen14 on heicjpg/acct3, probe_split_deepseek14 on
   heicfast/acct2) exercising the cross-GPU NF4 load path + the new stress
   gate on real hardware. Cell configs' probe_artifact_sha256 are updated
   to the landed artifacts BEFORE the cells launch; the factory fails
   closed otherwise. The kernels were made provably identical modulo the
   candidate identity lines (diff-verified).

5. Cell kernel allowlist extension: `src/traceguard/model_probe.py` added
   with justification (function-local import in the probe-model CLI handler
   only; grep-verified zero references across the scoring import graph —
   same precedent as smoke.py). All other attempt-3 files are
   docs/scripts/kaggle/tests prefixed (non-research by the frozen rule).

### 2026-10-27 — Split-layout admissions PASSED for both 14B monitors (deployment-scale stress gate on real hardware)

- `heicjpg/traceguard-probe-split-qwen14` (349 s): ALL 9 probe gates PASS on
  the cross-GPU NF4 load path. Stress prompt 18,140 chars (deployment max
  measured 15,659 — the gate exercises ~16% ABOVE deployment); peak reserved
  7.16 / 11.00 GiB per device vs 14.0 GiB budget. Resolved revision
  40c06982... == the frozen scale-cell revision (anchor continuity).
- `heicfast/traceguard-probe-split-deepseek14` (459 s): ALL 9 gates PASS;
  same stress scale; peak reserved 7.19 / 11.30 GiB. Resolved revision
  1df85071... == the frozen cross-cell revision.
- The asymmetric balance (accelerate places the larger share on device 1)
  is recorded; both devices hold >=3 GiB headroom at stress scale, and the
  allocator-hygiene env (expandable_segments) applies on top in the cells.
- Both cell configs' probe_artifact_sha256 pinned to the split artifacts;
  cell kernels' artifact_rel moved to runs/v6/probes/monitor_split/; factory
  fail-closed verification re-run locally for both cells: PASS.
- The stress gate would have REFUSED the attempt-2 single-GPU layout at
  admission (18,140 chars > the 15,615-char OOM boundary) — the structural
  gap is closed: capacity admission now measures deployment scale.

### 2026-10-28 — Scale cell (Qwen3-14B monitor) LANDED attempt-3: raw-arm recovery at 14B confirms capacity explanation; cross cell externally interrupted at 2/5 LLM conditions, re-launched

**Scale cell (`heicjpg/traceguard-mon-scale-qwen14`, v3 split layout, pinned
63b43cb, cell scale-monitor-qwen4-qwen14): COMPLETE, kernel_seconds 27050 (~7.5 h),
all critical rcs zero (monitor/analysis/canonicalize/verify/export). The
single_split worker scored 128/128 episodes on all 5 llm conditions with
ZERO generation failures (M2 invalid <= 2.3%, M2' retention >= 97.4%) —
the attempt-2 knife edge (C 18% OOM) is fully resolved by the dual-T4
split; the deployment-scale stress gate held in production.**

**Landing chain reproduced locally end-to-end**: verify re-grade 128/128,
canonicalize (env manifest ledgered), verify 0 problems, monitor gate
reproduced exactly (M1 PASS 1664 views/0 leakage; M3 FAIL raw arms 6/5/2/5
distinct vs late_fusion 26; M4 2 benign clusters; M6 0 allowed FPs — same
honest design facts as the anchor), analysis determinism PROVEN (only
output-path prefixes differ; all 1,664 score leaves identical), export
`release_complete: true`.

**Scientific result (exploratory) — the capacity-vs-instrument question is
resolved for the raw arm:**
- The identical v2 raw prompt that collapses to all-zeros at 4B yields a
  DISCRIMINATIVE score distribution at 14B (actions distinct=6,
  C distinct=5, fusion distinct=26) — the Phase-2-style raw-arm degeneracy
  is monitor-capacity-bound, NOT an unavoidable instrument property.
- AUROC (A / C): dev 1.0/1.0, holdout 1.0/0.9583, shift 1.0/1.0,
  validation 1.0/0.9643 — actions-only remains AT CEILING on this cohort,
  so positive ΔR(C−A) stays structurally blocked by benchmark action
  visibility (predeclared limitation, unchanged).
- RQ3 is now STRONGLY positive at capable scale: C decisively beats both
  content controls — E (shuffled CoT) collapses recall@FPR to 0.0 on all
  four splits while C stays 0.92–1.0, and D (noise) drops to 0.62 on dev.
  The 14B monitor genuinely uses reasoning CONTENT and ordering.
- Persuasion robustness improves with capability: the 4B anchor's
  holdout 1.0 → 0.0 CoT collapse does not reproduce at 14B (C 0.9167
  recall at 0 FPR). The authorization-laundering channel is attenuated
  but not eliminated at this scale (M3 shows raw arms still coarse).

**Cross cell (`heicfast/traceguard-mon-cross-deepseek14`): the attempt-3
run was externally cancelled at 27,050 s — the same second the scale cell
completed — after DeepSeek-R1-14B loaded (579 shards) and scored 2/5 llm
conditions (A and C, 128/128 each). Partial worker output archived at
`runs/v6/failed_attempts/cross_mon_interrupted_20261027/` (61 MB, full
provenance preserved). Per the no-compromise rule the cell is RE-RUN in
full (v3 pushed 2026-10-28, RUNNING); no partial score will be mixed into
the landed record.**

### 2026-10-28 — Headroom measurement (S5-decisive): the v6 action-visibility ceiling is UNIFORM — A-AUROC = 1.000 on every family and strategy at 14B

Per-family rank-based AUROC (Mann-Whitney, no threshold assumption; script
`scripts/headroom_by_family.py`, output archived at
`runs/v6/headroom_by_family_20261028.txt`) measured on the landed scale-cell
scores (128 episodes, 53 attempted attacks, 75 benign):

- 14B monitor, actions-only: **A-AUROC = 1.000 on all 4 families AND all 4
  attack strategies.** Even the v6 relational/non-action-visible family
  designs (cross-object refs, deferred effects) are fully action-inferable
  by a 14B semantic monitor — the dispositive information IS present in the
  action sequence for a capable reader, even though the syntactic rules
  monitor sees nothing (rules_v1 recall 0/53).
- 14B, actions+CoT: C-AUROC 0.958–1.000 — never above A (3 families
  marginally below). The CoT signal is real (E-AUROC drops to 0.73–0.92
  when shuffled) but redundant with action information.
- 4B v2 raw instrument: 0.500 everywhere (uniform degeneracy, confirming
  the capacity finding is uniform, not family-specific).

**S5 consequence (recorded per the preregistered estimand-integrity
policy):** the headroom blocking condition (#1) fails UNIFORMLY — recall
(actions) cannot enter the 0.40–0.70 band on this template inventory for
any capable monitor. A confirmatory-scale run (~550–650 episodes, ~40–60
GPU-h) on the current families would measure a structurally-null estimand
and is therefore NOT scientifically justified (no-compromise means not
burning quota on a premeasured null). The correct path: freeze the
exploratory matrix result as the v6 evidence, and predeclare the
confirmatory program as future work requiring genuinely intent-only
attacks (legitimate-appearing action sequences whose violation is
recoverable ONLY from reasoning), plus >=40 benign calibration clusters.
