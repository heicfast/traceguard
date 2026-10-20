# Developer review — TRACEGUARD v0.6.0 operations core

Reviewer: independent senior developer (25+ years: distributed systems, research
infrastructure), adversarial pre-experiment review.
Scope (read line-by-line, against `docs/AUTOMATION_AND_EVIDENCE_V6.md`):
`src/traceguard/workflow.py` (315 lines), `src/traceguard/errors.py` (68),
`src/traceguard/provenance.py` (128), `src/traceguard/cli.py` (workflow /
index-run / study-plan wiring), `configs/workflow_phase3.template.yaml`,
`tests/unit/test_research_operations.py`. Supporting modules consulted:
`ids.py`, `traces/storage.py`, `traces/schema.py`, `generation_runtime.py`,
`scripts/run_gpu_workers.py`, `scripts/pilot_gate.py`.
Repo state: `pyproject.toml` 0.6.0, HEAD `57a0c73`. Full test suite:
**166/166 passed**. Review method: line-by-line reading plus **22 executed
experiments** in `/tmp/tg_rev` (real commands, real journals, real signals:
SIGKILL, SIGINT, concurrent runs, byte-level journal surgery).

Display-hazard discipline: every journal/tamper question was settled with
programmatic byte-level assertions (`json.loads` per line, `b"..." in
open(p,"rb").read()`, exit codes, `pgrep`, inode/size checks). No conclusion in
this review relies on visually displayed bracketed text.

---

## VERDICT

**APPROVE-WITH-CONDITIONS** — for *attended pilot/exploratory* execution on
this core. The load-bearing integrity machinery is real and verified by
execution: hash-chained journal with full re-verification on every read,
atomic writes, digest-checked resume, working mutual-exclusion lock, strict
argv-only execution. **Do NOT run unattended/confirmatory workloads, and do
NOT wire scientific gates (`pilot_gate.py`, `monitor_gate.py`) into workflow
stages, until the conditions below are met.**

Counts: **1 BLOCKER, 8 MAJOR, 7 MINOR, 5 NOTE.**

Conditions (blocking unattended + confirmatory use):
1. Fix F1 (gate failures retryable into a pass; dead taxonomy wiring) before
   any gate script runs as a workflow stage.
2. Fix F3/F4 (unjournaled crash classes; dangling/overwritten attempt logs) —
   the failure contract in spec §3 must hold for *every* failure.
3. Fix F2 (stale-lock recovery) — today any hard crash bricks `resume` until a
   human deletes `workflow.lock`.
4. If any stage fans out subprocesses (the `run_gpu_workers.py` pattern), fix
   F8 (process-group kill on timeout) first.

---

## Findings

Severity, evidence (file:line), reproduction I actually ran, and the fix.

### F1 — BLOCKER: scientific-gate failures are retryable and can be retried into a pass; the failure taxonomy is dead code

Evidence:
- `workflow.py:275-287` — every non-timeout stage failure (which is how *every*
  gate failure manifests: `scripts/pilot_gate.py:20` "Exit code 0 iff the
  automatic gates pass") is classified `ErrorCode.SOFTWARE_DEFECT`, with
  `retryable = attempt <= stage.retries` (line 285). The retry loop
  (`workflow.py:228`) then re-executes the stage.
- `errors.py:18,20` — `STATISTICAL_GATE` and `USER_CANCELLED` are defined and
  never assigned anywhere in the codebase (grep across all `.py`: only the
  enum definition). `errors.py:49-65 classify_exception` — the function that
  *does* implement the ten-class taxonomy — is **never called** by
  `workflow.py`, `cli.py`, or anything else. `errors.py:24 RETRYABLE_CODES`
  correctly excludes `STATISTICAL_GATE`, but no code path can ever produce
  that code.
- Spec `AUTOMATION_AND_EVIDENCE_V6.md` §3: "Scientific gate failures are not
  transient and must never be retried into a pass by changing the threshold or
  data." The engine cannot express this: there is no per-stage "gate /
  non-retryable" declaration, and exit-code failures are always retried per
  `retries`.

Reproduction (E8, executed): a stage whose command emulates `pilot_gate.py`
semantics (exit 1 = predeclared threshold not met; exit 0 on a later
evaluation), `retries: 2`:

```
started attempt=1
 failed  attempt=1 code=software_defect retryable=True
started attempt=2
succeeded attempt=2          <- gate failure RETRIED INTO A PASS; final ok=True, CLI exit 0
```

The journal ends `ok:true` with no trace that a scientific verdict was
overturned by a retry. With any nondeterminism in the gated pipeline (GPU
sampling), attempt 2 can pass where attempt 1 legitimately failed.

Fix: (a) add a per-stage `gate: true` (or `retry_policy: never`) field;
gate stages get `retries=0` enforced and failures classified
`STATISTICAL_GATE`; (b) classify exit-code failures by stage role, not
"everything is a software defect"; (c) either wire `classify_exception` into
the engine or delete it; (d) pin with a test: a failing gate stage with
`retries: 2` must produce exactly one attempt and a non-retryable
`statistical_gate` envelope.

### F2 — MAJOR: stale lock permanently blocks resume after any hard crash

Evidence: `workflow.py:201-205` — `O_CREAT|O_EXCL` lock; no PID-liveness or
age check anywhere. `workflow.py:207` writes `pid=<n>` into the lock but
nothing ever reads it. `workflow.py:310-312` — `finally` unlink only runs on
clean unwinding; SIGKILL / OOM-kill / VM termination leaves the lock behind.

Reproduction (E6, executed): started a run, `kill -9`ed the orchestrator
mid-stage. `workflow.lock` survived; `traceguard workflow resume` then failed:
`RuntimeError: workflow lock already exists: .../workflow.lock` (exit 1) —
repeatedly. Only after manually `rm workflow.lock` did resume correctly
re-run the interrupted stage (journal `started`-without-`succeeded` was
handled properly).

Impact: the single most likely real-world event on a Kaggle box (session
death) bricks the engine's own recovery path, undocumented, on the first
resume attempt. Loud, not data-corrupting — but exactly the "lost runs" class
of failure this project has already suffered.

Fix: on `FileExistsError`, read the lock, check `os.kill(pid, 0)` / lock age;
if stale, take over (or at minimum print the exact recovery command). Add a
test: SIGKILL a run, resume must succeed without manual intervention.

### F3 — MAJOR: narrow `except` leaves common failures unjournaled, without failure envelopes, and without a status write

Evidence: `workflow.py:275` catches only
`(subprocess.TimeoutExpired, RuntimeError)`. Also `workflow.py:218-222`
(dependency-incomplete `RuntimeError`) is raised outside any envelope/journal
wrapper (see E16 below and M7).

Reproductions (all executed):
- E22 — argv[0] typo (`command: [traceguardd, --oops]`, `retries: 2`):
  `FileNotFoundError` from `subprocess.run` → orchestrator dies with a raw
  traceback; journal left with a dangling `started`, **no `failed` event, no
  envelope, `retries` ignored, no `workflow_status.json`**. Exit 1.
- E10 — stage emitting non-UTF-8 bytes (`sys.stdout.buffer.write(b'..\xff..')`):
  `UnicodeDecodeError` inside `subprocess.run(text=True)` → identical
  unjournaled crash.
- E16 — `continue_on_error: true` stage fails, dependent stage then hits
  `workflow.py:218`: `RuntimeError: stage dependent dependencies incomplete:
  ['broken']` — unjournaled, no envelope, no status file. (The
  `continue_on_error` feature is unusable for any stage with dependents.)

Spec §3 says "Every failure includes the stage, exception type, retryability
decision, attempt, and log path." These failure classes violate all five.

Fix: wrap each attempt in `except Exception as exc:`; classify via the
taxonomy (`FileNotFoundError`→`configuration`, non-retryable), journal a
`failed` event, keep the bounded-retry semantics. Handle
`continue_on_error` by marking dependents skipped-with-reason instead of
raising.

### F4 — MAJOR: timeout failures record a log path that is never written, and destroy the child's partial output; attempt logs collide across runs

Evidence:
- `workflow.py:240-251` — `atomic_write_text(log_path, process.stdout)` is
  *after* `subprocess.run`; on `TimeoutExpired` it never executes, and the
  handler at `workflow.py:275-287` discards `exc.stdout` (the process's final
  output — usually the most valuable hang diagnostics).
- `workflow.py:229` — log path is `logs/{stage}.attempt-{attempt}.log`, but
  `attempt` restarts at 1 in every `run_workflow` invocation
  (`workflow.py:228: range(1, stage.retries + 2)`), so a resume reuses the
  same file names as the failed run it resumes from.

Reproductions (executed):
- E7/E19 — stage sleeps past `timeout_seconds: 2`: journal gets `failed`
  events whose envelope references `logs/slowpoke.attempt-1.log`, but
  `logs/` is **empty**. The dying process's `PRE-TIMEOUT EVIDENCE` output
  (verified present in `TimeoutExpired.stdout`) is discarded.
- E9 — run 1 of a failing stage writes `EVIDENCE-FROM-INVOCATION-1` to
  `logs/failer.attempt-1.log`; resume runs attempt 1 again and **overwrites
  the same path** with `EVIDENCE-FROM-INVOCATION-2`. The journal now holds
  two `failed` events pointing at one file; run 1's forensic evidence is
  destroyed (byte-verified).

Fix: write the log inside the attempt loop from a `finally` (use
`exc.stdout` on `TimeoutExpired`); make log names unique per run (journal
sequence or timestamp prefix), never reuse a path already referenced by a
journal event.

### F5 — MAJOR: editing the workflow (changed stage command) never invalidates a skip — stale outputs from old code are silently reused

Evidence: `workflow.py:211-217` skip check uses only (a) latest `succeeded`
record for the stage name and (b) current output digests vs
`workflow.py:259-261` recorded digests. `workflow_sha256` is journaled on
`started` events (`workflow.py:236`) but **never consulted**, and `succeeded`
events don't even carry it.

Reproduction (E11, executed): ran stage `make` writing `'ok'`; rewrote the
workflow so `make` runs completely different logic; `resume` returned
`skip_verified` with the new `workflow_sha256` visible in the same output
(`c85eff…` → `e48aed…`). `result.txt` still contains the old run's `'ok'`.

Spec §2 is technically satisfied ("prior success record and every declared
output digest still match"), but the hazard the brief warns about — silent
stale stage-skip after a bugfix to stage code/args — is fully present, and
downstream stages consume outputs produced by a command that no longer
exists in the workflow file.

Fix: store `workflow_sha256` on `succeeded` events; on resume, re-run any
stage whose success record's digest differs from the current workflow digest
(or at least from the digest of that stage's command). Cheap, and the journal
already records the needed data.

### F6 — MAJOR: the shipped template's `outputs: []` stages create resume blind spots (monitoring / verify / doctor)

Evidence: `configs/workflow_phase3.template.yaml:15-20` (`monitoring`,
`outputs: []`), `:4-8` (`doctor`), `:27-31` (`verify`).
`workflow.py:148-156` — with no declared outputs, `_outputs_valid` reduces to
"a succeeded record exists" (vacuous truth), so resume never re-checks the
stage's *actual* side effects.

Reproduction (E20, executed): a `monitoring`-like stage (`outputs: []`)
wrote `scores.jsonl`; I deleted `scores.jsonl`; `resume` still returned
`skip_verified` and the measurement file was **not** regenerated. In the real
template, losing/corrupting `monitor_scores/` between runs is invisible to
`workflow resume` — the engine-side protection ends at the journal record,
and the only stages that would notice (`verify`, `index`) are themselves
no-output stages frozen on their own success records.

Fix: declare the monitoring stage's real outputs (score shard files are
predictable: `monitor_scores/<monitor>-<condition>-00001.jsonl`) or add
engine support for directory/glob outputs. Pure-check stages (doctor, verify)
should arguably always re-run — they are cheap.

### F7 — MAJOR: journal tail-truncation is undetectable ("tamper-evident" holds for retained records, not for the record set)

Evidence: `workflow.py:104-122` verifies `previous_event_sha256` linkage and
per-record digests of the lines *present*. There is no anchored head/commit
of the journal, so deleting the last event(s) leaves a perfectly valid chain.

Reproduction (E1c-1, executed): single-run journal
`[started, succeeded]`; byte-surgery removed the last line; `resume` accepted
the journal silently (no error) and re-ran the stage. Same for removing a
trailing `failed` event. (Deleting a *middle* record — E1c-2 — **is**
detected: `ValueError: workflow journal chain broken at event 1`, exit 1.)

Direction of failure is safe (truncation forces re-execution, never a silent
skip), but the spec's "tamper-evident events" claim is overstated, and an
operator can "lose" an inconvenient final failure event with zero evidence.

Fix: persist the journal head hash externally after each run (e.g., in
`workflow_status.json` / the run manifest / release checksums) and warn when
the journal is shorter than the last known head. At minimum, document the
limitation.

### F8 — MAJOR: `timeout_seconds` kills only the direct child — grandchildren (GPU worker processes) are orphaned

Evidence: `workflow.py:242-250` — `subprocess.run(..., timeout=...)` with no
`start_new_session=True` and no `os.killpg`; `subprocess.run` kills only the
direct child. `scripts/run_gpu_workers.py:99` fans out worker processes via
`Popen` — the natural stage command for scaled phase3 generation/monitoring.
(`generation_runtime.py:89,99` subprocess calls are short `git` probes, so
the *template as shipped* runs single long-lived children; the hazard
activates the moment a stage wraps workers.)

Reproduction (E7-t2, executed): stage command spawns a grandchild
(`time.sleep(25)`) then sleeps; `timeout_seconds: 3`. After the timeout the
journal correctly recorded `failed` (E7-t1: bounded 2 attempts, exit 1, direct
child dead — verified `pgrep` count 0), but the grandchild **survived**
(`pgrep -af` showed PID 9496 alive) until I killed it manually.

Impact on a single-GPU box: orphaned workers keep VRAM allocated and keep
writing into the run dir while the retried attempt starts — the classic
OOM death-spiral that burns the whole session quota.

Fix: `subprocess.Popen(..., start_new_session=True)`; on timeout,
`os.killpg(os.getpgid(pid), SIGKILL)` (with a short SIGTERM grace period);
same treatment in a top-level signal handler for orchestrator death.

### F9 — MAJOR (clustered with F3/F4): journal/CLI alarms are raw tracebacks — no envelope, no journal record of the tamper event itself

Evidence: every tamper/lock/config failure surfaces as an uncaught exception
through `cli.py:265-278` (`cmd_workflow` has no error handling) → Python
traceback, exit 1. Executed examples (E1c-2, E1d all four variants, E2 stale
lock, E12c missing workflow file): all exit 1 — the exit code is right, but
for an unattended fleet the operator gets no machine-readable failure object,
and `workflow_status.json` retains the *previous* run's contents (stale
status after a tampered-journal crash — observed).

Fix: catch at the CLI boundary, emit a `FailureEnvelope`
(`code=input_integrity` for tamper, `configuration` for validation), exit 1.

### MINOR findings

- **M1** `workflow.py:100-101` — `WorkflowJournal.__init__` does
  `mkdir(parents=True)`; the supposedly read-only `workflow status` **creates
  the work dir** (E12a, executed). Also fails on read-only mounts.
- **M2** `workflow.py:90-91` — `retries`/`timeout_seconds` are not validated:
  `retries: -3` makes the stage silently *never execute* (0 attempts, no
  journal events — E15c, executed); `timeout_seconds: -5` fails instantly as
  `transient_provider` (E-negto, executed). A typo produces nonsense
  behavior instead of a validation error.
- **M3** `workflow.py:279` — timeouts classified `TRANSIENT_PROVIDER` even
  for local GPU stages; retryable is plausible but the class is wrong, and
  the envelope's `retryable` contradicts `errors.py:24` semantics (see F1).
- **M4** `errors.py:20` — `USER_CANCELLED` never emitted: SIGINT during a
  stage leaves a dangling `started` and a `KeyboardInterrupt` traceback (E21b,
  executed; lock cleaned, child killed — no data hazard, but the taxonomy's
  cancellation class is fiction).
- **M5** `provenance.py:74-77` — provenance fields silently degrade: a run
  manifest without `experiment_stage`/`config_sha256`/`code_revision`
  yields an index with `null` values and `ok:true` (E13-P3, executed). For an
  evidence-provenance module, missing provenance should be a reported
  problem, not a null.
- **M6** `provenance.py:16-20,54-56` — `workflow.lock` is not excluded from
  the evidence index: a stale lock gets indexed, then the next run's normal
  lock cleanup makes `index-run --check` fail with a misleading
  `indexed file missing: workflow.lock` (E13-P1, executed).
- **M7** `workflow.py:218-222` — dependency-incomplete is a hard `RuntimeError`
  even when the cause is a `continue_on_error` failure (see E16 under F3);
  there is no "skipped-because-upstream-failed" journal event, so a resumed
  run can't distinguish "never attempted" from "attempted and failed" without
  correlating events.

### NOTE findings

- **N1** `cli.py:278` — `workflow status` always exits 0 (incomplete stages
  included); scripts cannot use the exit code as a completeness gate.
- **N2** `workflow.py:124-138` — every `append` rewrites and re-verifies the
  whole journal (inode churn verified, E18): atomic and safe, but O(n²); fine
  at stage-event granularity, would not be at episode granularity. "Append-only"
  is really "atomic full-rewrite" — the on-disk file is replaced each event.
- **N3** `provenance.py:97-125` — `index-run --check` verifies only files
  *listed* in the index; files added after indexing are invisible (E13-P2,
  executed: injected file, check still `ok:true`). By design per spec §4
  (`export` rebuilds), but operators should know `--check` is a point-in-time
  snapshot check, not a directory diff.
- **N4** `_successful_output_record` (`workflow.py:141-145`) accepts *any*
  prior success record: with multiple success records, deleting the newest
  still skips via an older one if digests match (observed in E1c-1 first
  variant). Spec-conformant; just be aware.
- **N5** CLI input errors at the config level (missing workflow file, missing
  run manifest, unknown assumption keys) are uncaught exception tracebacks
  (exit 1) rather than envelopes; Tier-0-acceptable, inconsistent with §3's
  "every failure" language.

---

## VERIFIED-CORRECT (executed or line-verified)

1. `workflow.py:61-67` — commands must be nonempty argv lists of nonempty
   strings; a shell-style string command is rejected (`ValueError: stage bad
   command must be a nonempty argv list`, E15a).
2. **No shell-string evaluation anywhere**: grep over `src/traceguard` finds
   no `shell=True`, no `os.system`, no `popen(`, no `shlex`. Executed probe
   E15b: `echo "hello; $(touch PWNED) && \`touch PWNED2\`"` ran with both
   substitution strings printed *literally*; no `PWNED` files created.
3. `workflow.py:68-73` — declared outputs confined to the workflow root
   (absolute paths and `..` rejected; E12 matrix).
4. `workflow.py:49-50,58-59,74-79` — schema version pinned; duplicate stage
   names, unknown dependencies, self-dependencies all rejected (executed
   matrix).
5. `workflow.py:104-122` — every journal read re-verifies the full chain:
   `previous_event_sha256` linkage plus per-record `sha256_json` digest.
   Executed (E1d): payload field edit → `digest mismatch at event 0`;
   prev-hash corruption → `chain broken at event 1`; forged appended event →
   `chain broken at event 2`; half-written line → JSON parse rejection. All
   exit 1 via both `resume` and `status`.
6. `workflow.py:124-138` + `traces/storage.py:35-57` — event hashing covers
   the previous hash; writes are durable-atomic (temp + fsync + rename +
   parent-dir fsync). A crash mid-write can only leave the *previous* intact
   file; torn lines are rejected on read (d4).
7. `workflow.py:202-205` — `O_CREAT|O_EXCL` lock gives real mutual exclusion
   (E5, executed: concurrent run B rejected with exit 1; **B did not unlink
   A's lock** — the `FileExistsError` raise at :204 precedes the
   try/finally at :310; journal contained exactly A's events; byte-verified).
8. `workflow.py:211-217,141-156` — skip requires success record **and** every
   declared output present with matching digest: unchanged resume →
   `skip_verified` with zero new journal events (E1a); edited output → re-run
   with output restored (E1b); deleted success record → re-run (E1c-1).
9. `workflow.py:228` — retries bounded (`range(1, retries+2)`): `retries: 1`
   produced exactly 2 attempts then gave up with `ok:false`, exit 1 (E7-t1).
10. `workflow.py:252-258` — both nonzero exit codes and missing declared
    outputs are journaled as `failed` with envelopes.
11. `workflow.py:259-271` — success records carry per-output SHA-256,
    duration, and log path; `started` events carry the full argv and
    `workflow_sha256` (useful provenance, even though the skip check ignores
    it — F5).
12. `errors.py:24` — `RETRYABLE_CODES = {resource_exhausted,
    transient_provider}` — the taxonomy itself correctly excludes
    `statistical_gate` and `data_quality` from retryability (the wiring, not
    the definition, is broken — F1).
13. `errors.py:36-46` — envelope payload carries code, message, stage,
    exception type, retryability, context (attempt + log), timestamp.
14. `provenance.py:54-90` — evidence index assigns every file
    path/role/bytes/sha256/media-type, hashes the ordered content list and
    the payload, and schema-validates before the atomic write (E13 build ok).
15. `provenance.py:107-125` — verify checks index self-digest, content-list
    digest, per-file existence + digest, and path escape. Executed: content
    mutation → `ok:false, indexed digest mismatch` (exit 1); crafted
    `../../outside` index entry → `path escapes run root` (exit 1); my own
    tampering of the index file was caught by both digest checks (E13-P4).
16. `cli.py:397-401,265-278` — `workflow plan/run/resume/status` wired as the
    spec §2 examples show; `run` and `resume` share semantics (spec §2),
    verified by E1a; `plan` is a true dry run: executes nothing, journals
    nothing (E12b).
17. `cli.py:281-299` — `index-run` build/`--check` dispatch correct; missing
    index → structured `{"ok": false, "problems": ["evidence index missing"]}`,
    exit 1.
18. `cli.py:243-252` — `study-plan` rejects unknown assumption keys
    (`ValueError: unknown planning assumptions: ['bogus_key']`, exit 1) and
    runs end-to-end on `configs/smoke.yaml` (exit 0).
19. Argparse contract: invalid subcommand/action/missing `--work-dir` →
    exit 2 with a clear message (E12c).
20. `configs/workflow_phase3.template.yaml` loads and plans cleanly
    (6 stages, dependencies resolve, all commands are argv lists; `late_fusion`
    is a real monitor — `monitor_runtime.py:92`).
21. Resume after interrupted stage (journal `started` without `succeeded`,
    lock manually removed) correctly re-runs the stage and completes (E6b).

---

## Experiment log (all executed in `/tmp/tg_rev`, real processes)

| # | Experiment | Result |
|---|---|---|
| E1a | Run tiny workflow; resume unchanged | `skip_verified`; journal unchanged (2 events) — PASS |
| E1b | Edit `result.txt`; resume | Stage re-ran; output restored; new started/succeeded — PASS |
| E1c-1 | Delete success record when it is the last line | Re-runs, **no tamper alarm** (truncation undetected) — F7 |
| E1c-2 | Delete success record from the middle | `ValueError: chain broken`, exit 1, raw traceback — detected |
| E1d | 4 corruption variants (field edit / prev-hash / forged append / half-line) | All exit 1 (digest mismatch / chain broken / chain broken / JSONDecodeError) — PASS on detection, F9 on reporting |
| E5 | Two concurrent `run`s | B rejected `workflow lock already exists` exit 1; A's lock preserved; journal = A's events only — PASS |
| E6 | SIGKILL orchestrator mid-stage, then resume | Stale lock blocks resume until manual `rm`; after `rm`, correct re-run — F2 |
| E7-t1 | Stage exceeds `timeout_seconds` | Direct child killed; bounded retries (2 attempts); `failed` events journaled; **`logs/` empty while envelopes reference log paths** — F4 |
| E7-t2 | Timeout with grandchild process | Grandchild **orphaned and alive** after timeout (PID verified) — F8 |
| E8 | Gate-style exit-1 stage, `retries: 2` | Failed attempt classified `software_defect`, `retryable: true`, **retried into a pass** (`ok:true`) — F1 |
| E9 | Failing stage across two runs | Both journal `failed` events reference the same `attempt-1` log path; run-1 log overwritten (byte-verified) — F4 |
| E10 | Stage emits invalid UTF-8 | Orchestrator crash `UnicodeDecodeError`; no `failed` event, no envelope, no status write; lock cleaned — F3 |
| E11 | Edit stage command; resume | Stage **skipped** despite new `workflow_sha256` — F5 |
| E12a | `status` on nonexistent work dir | **Creates the directory** — M1 |
| E12b | `plan` (dry run) | Executes nothing, writes no journal — PASS |
| E12c | argparse misuse | Exit 2, clear messages — PASS |
| E13 | `index-run` build/check/mutate/escape/lock/null-provenance | Mutation + escape + index-self-tamper detected (exit 1); lock file gets indexed then "goes missing" (M6); unindexed additions invisible (N3); provenance silently null (M5) |
| E14 | `pytest tests/` | 166/166 passed |
| E15a/b/c | String command / shell metacharacters / negative numbers | String rejected; metacharacters inert (no shell); `retries: -3` → stage silently never runs (M2) |
| E16 | `continue_on_error` + dependent | Unjournaled `RuntimeError: dependencies incomplete`, no status write — F3 |
| E17 | `status` exit code | Always 0, even with incomplete stages — N1 |
| E18 | Journal append mechanics | Full atomic rewrite per event (inode churn verified) — N2 |
| E19 | Timeout partial stdout | `TimeoutExpired.stdout` discarded; referenced log file absent — F4 |
| E20 | `outputs: []` stage, side effect destroyed | Resume still `skip_verified`; real output not regenerated — F6 |
| E21/21b | SIGINT during stage | Dangling `started`; no `user_cancelled` envelope; lock cleaned; child killed — M4 |
| E22 | argv[0] typo (`traceguardd`) | `FileNotFoundError` crash, unjournaled, `retries` ignored — F3 |

---

## Test-quality assessment (`tests/unit/test_research_operations.py`)

The three workflow tests pin **real invariants**, not just happy paths —
journal digest-mismatch on payload edit (`:83-90`), output-path escape
(`:93-101`), and skip-only-when-digests-match (`:65-80`, asserted via the
returned plan and journal event list). The evidence-index test
(`:49-62`) genuinely pins mutation detection. These are good tests.

But the riskiest semantics of this engine are **untested**: no test for
edit-output→re-run, delete-record→re-run, journal truncation (F7 is exactly
the gap), forged appends, chain-break vs digest-mismatch classes, lock
acquisition/conflict/staleness, timeout (log file existence! F4 would have
been caught by `assert log_path.exists()`), retry bounds, gate
classification, log-name collision across resumes, non-UTF-8 output,
`continue_on_error`, or empty-outputs semantics. The evidence-index verify
path has no path-escape test. Recommended additions (each is one tmp_path
test): the five resume/tamper cases from the spec §2 sentence, a timeout test
asserting the envelope's log file exists, a gate-stage retry prohibition test
(pairs with F1), and a stale-lock recovery test (pairs with F2).

---

## Spec-conformance summary (AUTOMATION_AND_EVIDENCE_V6.md)

| Spec claim | Status |
|---|---|
| §2 deterministic argv execution, no shell-string evaluation | PASS (verified) |
| §2 declared dependencies and outputs validated | PASS |
| §2 resumability: skip only on success record + all output digests | PASS as written; stale on workflow edits (F5) and empty-output stages (F6) |
| §2 lock prevents concurrent journal writers | PASS (E5); recovery path broken (F2) |
| §2 tamper-evident events | PARTIAL: content + chain verified; tail truncation undetected (F7); alarms unstructured (F9) |
| §2 per-attempt logs | FAIL (F4: missing on timeout, overwritten across resumes) |
| §3 stable failure classes | PARTIAL: 2 of 10 codes dead; exit failures misclassified (F1) |
| §3 every failure has stage/exception/retryability/attempt/log | FAIL for OSError/decode/dependency/cancellation classes (F3, M4); log path can dangle (F4) |
| §3 retries bounded in the workflow file | PASS |
| §3 gate failures never retried into a pass | **FAIL (F1, BLOCKER)** |
| §4 evidence index roles/digests/derivation contract; verify; export rebuild | PASS (provenance module solid; see M5/M6/N3) |
| §6 Tier 1 "journal tamper tests" | PARTIAL (one tamper mode of several) |

---

## Recommended fix order

1. F1 (gate non-retryability + taxonomy wiring) — small diff, protocol-critical.
2. F3 (broad except + envelope for every failure class) — small diff.
3. F4 (log capture in finally; unique log names) — small diff.
4. F2 (stale-lock liveness/age check) — small diff.
5. F8 (process-group kill on timeout) — small diff, big GPU payoff.
6. F5/F6 (digest-aware skip; declare monitoring outputs) — engine + template.
7. F7 (journal head anchoring), then the MINORs.
