# Security / evidence review — TRACEGUARD v0.6.0 evidence chain

Reviewer: independent senior security/evidence reviewer (25+ years, audit-grade
research-integrity engineering). Adversarial pre-experiment review of the NEW
v0.6.0 evidence chain (evidence index, index verification, release rebuild).

Scope (read completely): `docs/AUTOMATION_AND_EVIDENCE_V6.md` §4–6,
`src/traceguard/provenance.py` (new), `src/traceguard/verification.py`
(index integration, +31), `src/traceguard/release.py` (export + index rebuild,
+40), `src/traceguard/schemas/evidence_index.schema.json` (new),
`src/traceguard/traces/schema.py` (+5), `src/traceguard/workflow.py`
(journal hash-chaining / evidence-adjacent parts only; the developer review
owns the rest), `src/traceguard/traces/storage.py`, `src/traceguard/ids.py`,
`src/traceguard/cli.py` (verify/export/index-run wiring). Live archive used
for experiments: `runs/pre-tests-data/pilot/pilot_v0_run` (64 episodes,
privileged, 61+3 transactions/commits, 12 monitor-score shards, 5 reports,
checksums + artifact manifest). Repo state: v0.6.0, HEAD `57a0c73`,
`git status runs/` clean before and after review (all experiments ran on
byte-copies under `/tmp/tg6/`; the archive was never mutated).

Display-hazard discipline: the terminal eats `[m`, so **no** conclusion in
this review relies on visually displayed text. Every mutation was applied and
asserted programmatically (byte offsets, `json.loads`, `hashlib`, exit codes,
`sha256sum` cross-checks). E.g. the injected log line containing
`[monitoring]` was verified by `stat().st_size == 33` and by hashing, never by
reading terminal echo.

---

## 1. VERDICT

**APPROVE-WITH-CONDITIONS** — for *attended pilot (Tier 4) execution only*.

The index itself is genuinely good: deterministic, ordered, role-labeled,
schema-checked, and it catches naive content tampering at three layers
(self-digest, content digest, per-file digests), cross-anchored by the
checksum ledger and by the transaction/commit pair. `verify` checks the index
when present, and `export` rebuilds the index before writing final checksums,
exactly as §4 requires.

But the **release gate is fail-open against schema-valid tampering** (F1,
BLOCKER): `traceguard export` packages a corrupted archive as
`release_complete: true`, exit 0, and — by rebuilding — *destroys the prior
index/checksum tamper evidence*. Four further MAJOR gaps (additions invisible,
`artifacts/` strippable, measurement files unbound from their condition,
score↔episode binding hashes recorded but never verified) mean that the
"tamper-evident" promise currently terminates at git commits and the
transaction/commit records, not at the tool's own release gate.

**Conditions of approval:**

1. **Export embargo.** Until F1 is fixed, no `.tar.gz` produced by
   `traceguard export` on this build may be treated as certified evidence.
   Every packaged release must additionally pass `traceguard verify` *after*
   export (a post-export `verify` still catches F1's episode-tampering case
   via the transaction records — but it does **not** catch F1's
   deletion/addition laundering, hence the embargo), and the run directory
   plus `artifacts/` must be git-committed immediately after export.
2. **Pilot only.** No confirmatory/holdout (Tier 5) execution until F1–F5 are
   fixed, the fix is re-reviewed, and the tamper matrix in §4 is added to the
   Tier-1 test suite as regression tests (F13).
3. Consistent with the developer review: no unattended runs; keep
   `pilot_gate.py`/`monitor_gate.py` out of workflow stages.

---

## 2. Findings

Severity, location, the exact experiment that demonstrates it, and the fix.
All experiments ran against copies of
`runs/pre-tests-data/pilot/pilot_v0_run` in canonical v0.6 state (fresh
`export`: manifest → index → checksums; `verify` green, "evidence index
checked 156 files").

### F1 — BLOCKER: `export` re-certifies tampered archives and destroys prior tamper evidence

**Where:** `release.py:74-186` (`build_release` validation set + rebuild
order), `cli.py:173` (exit-code logic). Root cause: `build_release` validates
schema shape and id-set equality only; it never runs the `verify_run`
battery, never compares the rebuilt index/ledger against the ones it
overwrites, and regenerates `checksums.sha256` + `evidence_index.json`
unconditionally.

**Experiments (all on `/tmp/tg6` copies):**

- *k_ship* — one surgical byte changed in `episodes/shard-00001.jsonl`
  (`propose_patch` → `qropose_patch`; asserted `n_bytes_diff == 1`, JSON
  parse + envelope schema still valid). `verify` before export: **rc 1**
  (`transaction/envelope mismatch`, `1 checksum mismatches`,
  `indexed digest mismatch`). `export --archive ...`: **rc 0**, output
  `ok: true, release_complete: true, validation_errors: []`, and a **7.17 MB
  corrupted `k_ship.tar.gz` was shipped to disk**. Post-export `verify` on the
  shipped run dir: the checksum-mismatch and index problems are *gone*
  (ledger + index were rebuilt over the tampered content); only the
  transaction/commit cross-check still fires.
- *a_byteflip / a_surgical* (matrix rows a1/a2): same outcome without
  `--archive` — export rc 0, `release_complete: true`.
- *c_delfile* — `reports/fig_scores_holdout.png` deleted: pre-export checks
  catch it (index "indexed file missing"; ledger "1 checksum mismatches");
  `export` then **rc 0** silently packages a release without the PNG
  (fresh index 155 files), and **post-export `verify` is fully green** — the
  deletion is completely laundered; nothing in-tool detects it anymore.
- *b_addfile* — an injected `logs/generation.log` is silently incorporated
  into the rebuilt index (role `execution_log`) and ledger.

**Why it matters:** the export gate is the last line of defense and the one
command whose purpose is fail-closed certification ("Research-stage releases
fail closed", `release.py:83-84`). Here it certifies tampered monitor-input
content as a complete release and simultaneously erases the tamper evidence
the earlier layers had produced. Downstream, only the `transactions/` +
`commits/` records (which pin the original envelope/privileged digests and
which export does not rewrite) still expose the k_ship corruption — an
attacker who also regenerates those (they are plain JSON files;
`record_episode` is not required to forge a consistent pair) leaves the
shipped release fully self-consistent. `ACADEMIC_METHODS_V6.md` §6 gate 8
("evidence-index and checksum verification") is a precondition for
*confirmatory claims*; this build cannot enforce it at the release gate.

**Fix (small, high-leverage):**
1. `build_release` must run the full `verify_run` battery *before* writing
   anything and refuse (nonzero exit, structured failure) on any problem —
   at minimum the transaction/commit/envelope/privileged cross-checks
   (`verification.py:126-149`), manifest↔episode `config_sha256`
   consistency (`verification.py:56-62`), privileged `events_audit` hash
   checks (`verification.py:79-97`), and duplicate score-id detection.
2. If `artifacts/checksums.sha256` or `artifacts/evidence_index.json` already
   exist, validate them against current content *before* overwriting, and
   record the prior `index_sha256`/ledger digest in the new release manifest
   so the rebuild is auditable, never silent.
3. Add an `export` stage with `needs: [verify]` to
   `configs/workflow_phase3.template.yaml` (today export is not in the
   workflow at all, so the orchestrated verify→index ordering does not gate
   it) or a `--require-verified` default.

### F2 — MAJOR: unindexed file additions are invisible to the entire chain

**Where:** `provenance.py:97-125` (`verify_evidence_index` iterates only the
*indexed* list — it never enumerates the directory), `verification.py:174-188`
(checksum check is also one-directional: lines in the ledger are checked;
files on disk not in the ledger are not), `verification.py:190-194`.

**Experiment (matrix row b):** added `logs/generation.log` (33 bytes,
content-verified by hash) to a canonical copy → `index-run --check` rc 0,
`verify` rc 0 (no problems, no warnings), `export` rc 0 (then launders the
file into index + ledger, see F1). Contrast row b2: adding a duplicate
episode shard *is* caught (`duplicate episode ids present`, export rc 1) —
so the gap is specifically non-episode additions: injected logs, planted
"reports", stray root files.

**Fix:** `verify_evidence_index` must `rglob` the run dir, subtract the three
design exclusions (`provenance.py:16-20`), and report any on-disk file absent
from the index ("unindexed file present"); `verify_run` must additionally
require exact set-equality between the ledger and the on-disk file set.

### F3 — MAJOR: `artifacts/` is strippable — `verify` stays green with zero tamper-evidence metadata

**Where:** `verification.py:174-194` ("when present" semantics for both
checksums and index); contrast `docs/AUTOMATION_AND_EVIDENCE_V6.md` §4
(`artifacts/` = required content: release manifest, evidence index, checksum
ledger) and `ACADEMIC_METHODS_V6.md` §6 gate 8.

**Experiment (matrix row g):** `rm -rf artifacts/` on a canonical copy →
`traceguard verify` **rc 0, `ok: true`, zero problems and zero warnings**
(the only signal comes from `index-run --check`, rc 1 "evidence index
missing", which nobody is required to run). `export` then rc 0 rebuilds
everything. An attacker — or an accidental copy with `artifacts/` excluded —
passes the main integrity gate with all anchors removed.

**Fix:** in `verify_run`, read `experiment_stage` from `run_manifest.json`;
for `{pilot, confirmatory, online}` require `artifacts/evidence_index.json`
and `artifacts/checksums.sha256` to exist and verify (hard problem, not a
warning). Keep the lenient path only for `smoke`/`gpu_smoke`.

### F4 — MAJOR: monitor measurements are not bound to their shard (condition/monitor filename)

**Where:** analysis reads scores by filename glob — `traces/storage.py:276-283`
(`load_scores(monitor, condition)`), `analyses/pair.py:57`; but no check
anywhere compares a record's `condition`/`monitor_name`
(`schemas/monitor_score.schema.json:8`, 18-19) to the shard it lives in:
`verification.py:151-171` checks schema + score-id uniqueness + view_hash
presence only; `release.py:104-132` checks schema + raw-audit fields only.

**Experiment (matrix rows f_swap_role / f2_launder):** swapped the contents
of `monitor_scores/rules_v1-actions-00001.jsonl` and
`monitor_scores/rules_v1-actions_plus_noise-00001.jsonl` (byte-level
`read_bytes`/`write_bytes` swap, asserted). Pre-export: caught by per-file
digests (index + ledger). But `export` **rc 0**, post-export `verify` **rc 0,
fully green**, and the shipped archive now serves
`load_scores('rules_v1','actions')` → records whose `condition` is
`actions_plus_noise` (programmatically asserted: the returned condition set
is `{'actions_plus_noise'}`). Downstream analysis would compute the `actions`
arm's statistics from `actions_plus_noise` measurements — wrong science,
shipped as a verified-complete release.

**Fix:** in both `verify_run` and `build_release`, parse
`{monitor}-{condition}-{shard}.jsonl` from the filename and assert every
record's `monitor_name`/`condition` matches; additionally recompute
`score_id == monitor_score_id(episode_id, condition, monitor_name,
prompt_hash)` (`ids.py:73-77`) per record.

### F5 — MAJOR: the score↔episode binding is *recorded* but never *verified* (dead-weight hashes)

**Where:** `view_hash` — produced in `monitors/interface.py:48-49`,
`traces/views.py:67`; checked only for *presence* at `verification.py:170`
and `release.py:119`. `prompt_hash` — presence only (`release.py:115-125`).
`score_id` — uniqueness only (`verification.py:163-169`). And
`traces/storage.py:286-295` `envelope_fingerprint()` — the one function that
fingerprints the observable episode content for binding — is **dead code**:
grep shows only its definition and `__all__` entry; no caller exists
anywhere in `src/`.

**Experiment/derivation:** because nothing recomputes the view from the
stored envelope, the a2/k_ship episode tamper (F1) is *not* contradicted by
any monitor score record: the scores still carry the original episodes'
`view_hash` while the archived `episodes/` content no longer produces it, and
no tool notices. The measurement provenance chain ("this score was produced
from exactly this monitor-visible input") is asserted, not checked — this is
precisely "recorded with a hash that isn't verified anywhere".

**Fix:** in `verify_run`/`build_release`, for each score record rebuild the
condition view from the stored episode envelope (the `traces/views.py`
machinery) and require `sha256_text(view.text) == rec["view_hash"]`; wire
`envelope_fingerprint` into the score records at creation time and into the
audit at verification time; recompute `prompt_hash` from the rendered prompt.

### F6 — MINOR: build/verify symlink asymmetry — `index-run` builds an index its own check rejects

**Where:** `provenance.py:54-66` (`rglob` + `is_file()` follow symlinks; no
`is_symlink` guard) vs `provenance.py:115-119` (`resolve()` + `relative_to`
escape check).

**Experiment (matrix row h):** `notes.txt -> /etc/hostname` inside the run
dir: `index-run` (build) happily indexes it (entry `path: notes.txt`, role
`run_metadata`, digest of `/etc/hostname`), then `index-run --check` **rc 1**
`"path escapes run root: notes.txt"`. So an export on such a tree ships an
archive whose own `verify` fails.

**Fix:** hard-fail on symlinks in `build_evidence_index` (and in
`write_checksums`), or resolve-and-require containment at build time.

### F7 — MINOR: malformed checksum ledger crashes `verify` with a raw traceback

**Where:** `verification.py:179` — `digest, rel = line.split("  ", 1)`
raises `ValueError` on a line without the double-space separator.

**Experiment (matrix row i):** appended `garbage-no-separator-line` to
`checksums.sha256` → `verify` rc 1 (fail-closed) but as an uncaught
`ValueError: not enough values to unpack` traceback, not a structured
problem; violates the failure-envelope contract (`errors.py`) and matches the
developer review's F9 family. Note also `verification.py:180-181` accepts
ledger paths like `../../x` without an escape check (read-only impact).

**Fix:** parse defensively (`partition("  ")`), count bad lines as checksum
problems, reject paths escaping the run root, emit a `FailureEnvelope`.

### F8 — MINOR: checksum-ledger exclusion is name-based, not path-based

**Where:** `traces/storage.py:247` — `path.name != "checksums.sha256"`
excludes *any* file with that name anywhere in the tree.

**Experiment:** created decoy `logs/checksums.sha256` in a copy, ran
`export` (rc 0): the decoy **is** in the evidence index (role
`execution_log`) but is **not** in the ledger (programmatically checked both).
A tamperable file the ledger never covers.

**Fix:** exclude exactly `artifacts/checksums.sha256` by relative path
(mirror `EXCLUDED_INDEX_PATHS`).

### F9 — MINOR: `verify` mutates the audited tree

**Where:** `traces/storage.py:103-105` — `RunStore.__init__` does
`mkdir(parents=True, exist_ok=True)` for seven directories.

**Experiment:** `traceguard verify --run-dir /tmp/tg6/m_empty` (nonexistent)
→ rc 1, but the command first created `artifacts/ commits/ episodes/ logs/
monitor_scores/ privileged/ transactions/` in the target. A read-only audit
command must not write to the evidence tree.

**Fix:** read-only construction for verification paths (lazy/optional
`mkdir`).

### F10 — MINOR: canonical ordering unenforced at verify time; export surfaces a non-reproducible fingerprint

**Where:** `provenance.py:112` recomputes the content digest over the
*stored* order (no `sorted` check); `release.py:216` + `cli.py:163` print
`index_sha256` as "the" evidence fingerprint.

**Experiment (matrix rows e/e2):** reversing the `files` array and
recomputing both internal digests → `index-run --check` **rc 0** (reorder is
internally consistent); the inventory hash `content_index_sha256` *does*
change (asserted), so order matters to the fingerprint, but nothing requires
the canonical sort at verify time. Separately: two builds over identical
content produced different `index_sha256`
(`a1f03a1a…` vs `1d157e7e…`) because it covers `generated_at_utc`
(`provenance.py:73,90`), while the reproducible inventory hash
`content_index_sha256` was byte-identical across three builds — yet export
prints only the timestamp-dependent one, so operators will log a value that
changes on every rebuild of identical content.

**Fix:** in `verify_evidence_index`, require `[f["path"] for f in files] ==
sorted(...)` and compute the content digest over the sorted list; have
export print `content_index_sha256` alongside (or instead of)
`index_sha256`.

### F11 — NOTE: index metadata is not re-bound to `run_manifest.json`

`provenance.py:74-77` copies `run_id`/`experiment_stage`/`config_sha256`/
`code_revision` from the run manifest; `verify_evidence_index` never compares
them back. Experiment (row j): index `run_id` forged to
`totally-different-run` with recomputed digests → `index-run --check` rc 0;
only the checksum ledger caught it in `verify`. Fix: bind these fields
against `run_manifest.json` during verification (note `verify_run` already
binds `config_sha256` between manifest and episodes at
`verification.py:56-62`).

### F12 — NOTE: problem lists are silently truncated

`provenance.py:125` caps at 100 problems, `verification.py:200` at 50. A
tamper event producing more mismatches hides detail; print the true count.

### F13 — NOTE: Tier-1 test coverage for the new index is one happy path + one mutation

`tests/unit/test_research_operations.py:49-62` covers build → verify-ok →
mutate one file → verify-fail on a 2-file synthetic run. No tests for
additions (F2), deletions, index self-digest/JSON tampering, reorder,
symlinks, export integration, or determinism. The §6 Tier-1 claim ("journal
tamper tests") holds for the journal but is only marginally realized for the
evidence index. The tamper matrix in §4 of this review should become the
regression suite.

### F14 — NOTE: journal tail-truncation is undetectable (independently confirms developer review F7)

`workflow.py:104-122` validates the chain over the *retained* prefix;
dropping the last journal events leaves a fully valid chain. Experiment: a
4-event journal truncated to 2 events → `records()` returns 2, no alarm;
deleting a *middle* event is caught (`"workflow journal chain broken at
event 1"`). In scope for me only as confirmation; fix tracked by the
developer review (external head anchor, e.g. head digest in the run
manifest).

### F15 — NOTE: media type for `.jsonl` is `application/octet-stream`

`provenance.py:64` uses `mimetypes.guess_type`, which does not know `.jsonl`
(all 20 `.jsonl` files in the pilot index are `application/octet-stream`).
Harmless but the index would be more useful to third-party auditors with
`application/x-ndjson` registered.

---

## 3. Verified correct (with file:line)

1. `provenance.py:47-94` + `schemas/evidence_index.schema.json` — index
   builds on the real 158-file archive: 156 entries (158 minus the 3 design
   exclusions at `provenance.py:16-20`; `artifacts/` here held
   `checksums.sha256` + `artifact_manifest.json`), every entry carries
   `path`, `role`, `bytes`, `sha256`, `media_type`; paths unique; schema
   validation at build time (`provenance.py:91`).
2. Determinism: `content_index_sha256` byte-identical across three rebuilds;
   `files` array is sorted (`provenance.py:54`) and the inventory hash covers
   the ordered list (`provenance.py:70,81`). (Caveat on the *surfaced*
   fingerprint: F10.)
3. Digest correctness: per-file SHA-256 and byte counts match the filesystem;
   independently cross-checked with the external `sha256sum` tool for
   `episodes/shard-00001.jsonl`.
4. Role assignment matches the §4 run-directory contract exactly:
   `episodes/` → `monitor_visible_episode` (4), `privileged/` →
   `privileged_ground_truth_and_raw_inference` (4), `monitor_scores/` →
   `monitor_measurement_and_raw_inference` (12), `transactions/` →
   `write_ahead_transaction` (64), `commits/` → `transaction_commit_marker`
   (64), `reports/` → `derived_analysis` (5), root manifests/ledger →
   `run_metadata` (3); the embedded `derivation_contract`
   (`provenance.py:83-88`) matches the doc's ground-truth / monitor-input /
   measurement / derived partition.
5. `verification.py:190-194` — `verify` checks the index when present (spec
   §4): pristine canonical copy warns `evidence index checked 156 files`;
   every index problem propagates into `verify` results (rows a1, c, d).
6. `release.py:183-186` — export ordering is exactly per spec §4:
   `artifact_manifest.json` → `build_evidence_index` → `write_checksums`;
   the final ledger (158 lines) covers both `artifacts/evidence_index.json`
   and `artifacts/artifact_manifest.json` (verified programmatically).
7. Layered tamper detection for naive attacks: single-byte episode flip →
   `indexed digest mismatch` (index) + `transaction/envelope mismatch` +
   checksum mismatch (row a1); index-JSON hash edit → *all three* index
   checks fire: self-digest, content-list digest, per-file digest (row d);
   cross-role content swap → per-file digests + episode-schema rejection +
   transaction mismatch (row f); attacker-recomputed index digests still
   caught by the checksum ledger in `verify` (rows d2, e2, j).
8. `verification.py:126-149` — the transaction/commit/envelope/privileged
   cross-checks are the strongest in-tool anchor: they detect episode
   tampering *even after* export destroyed the index/ledger evidence
   (k_ship post-ship verify: `transaction/envelope mismatch:
   ep-305671c64f21c808c7a565cb`).
9. `provenance.py:115-119` — path-escape check (`resolve()` +
   `relative_to`) correctly flags symlinked index entries pointing outside
   the run root (row h).
10. Fail-closed on structurally broken input: schema-invalid episodes↔
    privileged swap → `export` rc 1 via `ValueError`, **no archive written,
    no temp leftovers** (`/tmp/tg6/out/.tmp-release-*` empty); duplicate
    episode shard → export rc 1 (`release.py:94-96`); missing run manifest →
    build/verify/`index-run` all fail.
11. `workflow.py:104-138` — journal chain: full re-verification on every
    read (append revalidates the whole chain; genesis `"0"*64`);
    middle-event deletion caught; events hash
    `previous_event_sha256` links (`workflow.py:112-121`); atomic rewrite via
    `atomic_write_text`.
12. `ids.py:21-37` — canonical JSON (`sort_keys`, compact separators,
    `ensure_ascii`, `allow_nan=False`) makes all digest computations
    deterministic; `traces/storage.py:35-57` — atomic writes with fsync of
    file *and* parent directory.
13. `configs/workflow_phase3.template.yaml:27-38` — the orchestrated path
    gates `index` behind `verify` (partial mitigation for F1; export is not
    a workflow stage, which is why the condition in §1 is needed).
14. Existing Tier-1 evidence tests pass (10/10 across
    `tests/unit/test_research_operations.py` and `test_evidence_v2.py`).

---

## 4. Tamper-experiment matrix (full results)

Baseline: canonical copy of the real pilot archive
(`/tmp/tg6/base`): `verify` rc 0 (`ok: true`, warnings: outcome re-grade 64
episodes, evidence index checked 156 files), `index-run --check` rc 0,
`export` rc 0. "Caught" = nonzero exit with the named problem.

| # | Tamper (applied to a fresh copy) | `index-run --check` | `verify` | `export` |
|---|---|---|---|---|
| a1 | 1 byte XOR @ offset 219224 of `episodes/shard-00001.jsonl` (all JSON lines still parse) | **CAUGHT** rc 1: `indexed digest mismatch: episodes/shard-00001.jsonl` | **CAUGHT** rc 1: `transaction/envelope mismatch`, `1 checksum mismatches`, index digest mismatch | **NOT CAUGHT** rc 0, `release_complete: true` |
| a2 | 1-char surgical edit (`propose_patch`→`qropose_patch`; JSON + envelope schema valid, asserted `n_bytes_diff==1`) | **CAUGHT** rc 1 (same as a1) | **CAUGHT** rc 1 (same trio) | **NOT CAUGHT** rc 0, `release_complete: true` |
| b | Added unindexed `logs/generation.log` (33 B) | **NOT CAUGHT** rc 0 | **NOT CAUGHT** rc 0 (no warning) | rc 0 — silently incorporated into rebuilt index (role `execution_log`) + ledger |
| b2 | Added duplicate episode shard `episodes/extra.jsonl` | rc 0 (index stale) | **CAUGHT** rc 1: `duplicate episode ids present` | **CAUGHT** rc 1 `ValueError: release validation failed: duplicate episode_id` |
| c | Deleted indexed `reports/fig_scores_holdout.png` | **CAUGHT** rc 1: `indexed file missing` | **CAUGHT** rc 1: `1 checksum mismatches` + `indexed file missing` | **NOT CAUGHT** rc 0 — PNG silently dropped (index 155 files); **post-export `verify` rc 0, fully laundered** |
| d | Index JSON: one hex char of `episodes/shard-00001.jsonl`'s sha256 changed (single occurrence, asserted) | **CAUGHT** rc 1 ×3: self-digest mismatch, content-list digest mismatch, indexed digest mismatch | **CAUGHT** rc 1 (ledger + all three index problems) | rc 0 (rebuilds — see F1) |
| d2 | d + attacker recomputes `content_index_sha256` + `index_sha256` | **CAUGHT** rc 1: `indexed digest mismatch` (per-file hash still wrong) | **CAUGHT** rc 1: `1 checksum mismatches` (ledger is the anchor) | rc 0 |
| e | `files` array reversed + internal digests recomputed | **NOT CAUGHT** rc 0 (order unenforced); inventory hash `content_index_sha256` **changed** (asserted) | **CAUGHT** rc 1: `1 checksum mismatches` | rc 0 |
| f | Contents swapped: `episodes/shard-00001.jsonl` ↔ `privileged/shard-00001.jsonl` (cross-role) | **CAUGHT** rc 1: two `indexed digest mismatch` | **CAUGHT** rc 1: envelope schema violation, 16 ledger entries without episodes, transaction mismatches | **CAUGHT** rc 1 (schema `ValueError`); no tar, no temp leftovers |
| f2 | Contents swapped: `rules_v1-actions` ↔ `rules_v1-actions_plus_noise` score shards (same role, different measurement condition) | **CAUGHT** rc 1: two digest mismatches | **CAUGHT** rc 1: `2 checksum mismatches` + index | **NOT CAUGHT** rc 0; **post-export `verify` rc 0**; `load_scores('rules_v1','actions')` now returns `{'actions_plus_noise'}` records → misattributed measurements shipped (F4) |
| g | `artifacts/` deleted entirely | **CAUGHT** rc 1: `evidence index missing` | **NOT CAUGHT** rc 0 — `ok: true`, zero problems, zero warnings (F3) | rc 0 (rebuilds all) |
| h | Symlink `notes.txt → /etc/hostname`; index rebuilt | build indexes it (digest of `/etc/hostname`); **check CAUGHT** rc 1: `path escapes run root` (F6 asymmetry) | rc 0 (pre-rebuild state) | rc 0 |
| i | Malformed line appended to `checksums.sha256` | rc 0 (not its scope) | rc 1 but **crash**: `ValueError: not enough values to unpack` (F7) | rc 0 (rebuilds ledger) |
| j | Index `run_id` forged (`totally-different-run`) + digests recomputed | **NOT CAUGHT** rc 0 (F11) | **CAUGHT** rc 1: `1 checksum mismatches` (ledger only) | rc 0 |
| k_ship | a2 + `export --archive /tmp/tg6/out/k_ship.tar.gz` | (verify pre-export rc 1) | (see row) | **rc 0, `ok: true`, `release_complete: true`, corrupted 7,172,397-byte tar shipped**; post-ship `verify` rc 1 reduced to the single `transaction/envelope mismatch` |
| — | Journal: tail truncation (4→2 events) | n/a | n/a | n/a — **undetected** (`records()` returns 2, no alarm; F14) |
| — | Journal: middle-event deletion | n/a | n/a | n/a — **CAUGHT**: `workflow journal chain broken at event 1` |
| — | Decoy `logs/checksums.sha256` + export | in index (`execution_log`) | ledger silent about it | rc 0 — file covered by index but **never by ledger** (F8) |
| — | `verify` on nonexistent run dir | n/a | rc 1, but **created 7 directories** in target (F9) | n/a |

Coverage summary (checksum-coverage question): the index covers every file
in the run dir except the three `artifacts/` integrity files
(`checksums.sha256`, `artifact_manifest.json`, `evidence_index.json`) —
`logs/`, `commits/`, `transactions/`, root manifests are all covered; the
ledger in turn covers the index and the release manifest. So the in-tree
trust chain is: per-file digests → `evidence_index.json` /
`artifact_manifest.json` → `checksums.sha256` → (nothing in-tree) → git
commit + workflow journal. That final hop must stay mandatory procedure.

Provenance dead-weight audit (question as posed): dead/verified-only-by-
presence hashes — `view_hash`, `prompt_hash`, `score_id` (F5);
`envelope_fingerprint` never called (F5); index metadata fields unbound
(F11). Genuinely verified: envelope/privileged digests pinned in
transactions and re-checked (`verification.py:126-149`),
`transaction_sha256` in commits, `rendered_prompt_hash`
(`release.py:126`), monitor `inference_audit` embedded hashes
(`release.py:128-132`, `verification.py:85-97` for privileged), the three
index digests. Critical but NOT recorded anywhere: nothing pins the *prior*
ledger/index digests when export overwrites them (F1), and nothing records a
journal head anchor (F14).

---

## 5. Next actions (ordered)

1. Fix F1 (verify-before-rebuild + preserve prior anchor digests in the
   release manifest) — small diff, unblocks the export embargo.
2. Fix F2 (reverse enumeration) and F3 (require artifacts for research
   stages) — both are contained in `provenance.py`/`verification.py`.
3. Fix F4/F5 (bind score records to shard filename and to episode content
   via recomputed views; wire in `envelope_fingerprint`).
4. Port the §4 matrix into `tests/unit/test_research_operations.py` as
   parametrized regression tests (F13), including the two builds → equal
   `content_index_sha256` determinism assertion.
5. Minor hardening: F6 symlink guard, F7 defensive ledger parse, F8
   path-based exclusion, F9 read-only verify, F10 sorted-order enforcement +
   surface `content_index_sha256`.
6. Re-run this review's matrix against the fixed build before the
   preregistration freeze; only then lift the pilot-only condition.
