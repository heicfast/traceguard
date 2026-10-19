# QUALITY MANUAL — TRACEGUARD

Version 1.0 · enforceable from the first commit. Every agent (human or automated) executing TRACEGUARD work is bound by this manual. Violations are recorded in `docs/deviations.md` and block gate progression.

## 1. No-fallback policy

**Definition:** a fallback is any silent substitution of a weaker method, cheaper model, looser check, or fabricated value when the intended method fails.

**Rules:**

| ID | Rule | Enforcement |
|---|---|---|
| NF-1 | A failed inference step is recorded as a failure (`runtime_error` episode). It is NEVER retried with a different model, temperature, or prompt to "make it work" within the same stage. | `generation_runtime` raises; runner records status; analysis censors with reason |
| NF-2 | Fake/simulated model outputs are admissible ONLY in `smoke` stage and unit tests. The analysis and any research-stage runtime hard-refuse providers named `fake*`. | `build_provider` raises for research stages; `EpisodeRecord` carries provider id |
| NF-3 | An invalid monitor score is recorded as invalid with its raw reason. It is never imputed, averaged over, or retried silently. | `MonitorScore.valid`, `CensorLog` |
| NF-4 | If a dependency is missing (GPU, torch, model download), the run stops with an explicit error. No CPU mock is silently substituted for a GPU result. | provider constructors raise; no except-and-continue |
| NF-5 | Missing reasoning traces are recorded as missing (condition C marks the episode missing for paired analysis). No retrospective explanation, re-prompting, or reconstruction is permitted. | `trace_available` flag; paired censoring |
| NF-6 | If a threshold/calibration cannot be computed (insufficient benign data), the metric is reported as uncomputable with the reason — never quietly switched to a different, easier quantity without a preregistered rule. | `precision_gate` in analysis |
| NF-7 | Exceptions are never swallowed without a log entry. Every `except` either re-raises, records a typed error record, or appends to the censor log. | code review + `verify_run` audit |

## 2. No-hallucination policy

**Definition:** a hallucination is any claim, number, or artifact presented as evidence that is not derived from recorded, hash-linked artifacts.

| ID | Rule | Enforcement |
|---|---|---|
| NH-1 | No number may appear in any report unless it is computable from stored episode/score records by a committed script. | analysis reads only `runs/`; reports carry config + data hashes |
| NH-2 | Ground-truth labels are produced only by `sandbox.grader` from state transitions. Model text is never used as evidence of outcome labels. | grader is the sole label source; verification re-grades independently |
| NH-3 | Reasoning traces stored in evidence are byte-exact model outputs (whitespace-newline strip only, as per the Qwen3 model card). No editing, translation, summarization, or repair. | `split_reasoning`/token-split preserve raw; raw output retained in privileged audit |
| NH-4 | Prompts committed in the repo are the prompts actually used (hash recorded per score). Any prompt edit after freeze is a deviation and invalidates affected results. | `prompt_hash` in every `MonitorScore` |
| NH-5 | Documentation must not describe features as "working" that fail tests; status tables must reflect the actual test/gate state. | README status table tracks CI outputs |
| NH-6 | External claims about prior work must cite a primary source (paper, post, repo) — no invented findings. | `docs/prior_art_matrix.md` with links |
| NH-7 | An agent that cannot verify a fact must either verify it (test, run, fetch) or mark it UNVERIFIED in the worklog. | worklog reviews |
| NH-8 | TOOL-CHANNEL ARTIFACTS: agent tooling display layers may silently rewrite or elide bracket-like and tag-like text (observed: `[h...` sequences eaten in displayed output; a content filter rewrote think-tag literals in Write-authored source files in a prior session). Therefore (a) NO verification may rely on visual inspection of bracketed content — every check must be a programmatic assertion; (b) files carrying tag-like literals must be built from `chr()` fragments or verified byte-exact after write; (c) source literals are guarded by tests (e.g. `test_think_tag_constants_byte_exact`). | codepoint-level checks; canary writes; byte-exact regression tests |

## 3. Verification and validation cadence

**Per change (every commit):**
1. `python -m pytest -q` — full suite green (121 tests).
2. `python -m ruff check src tests` — lint clean.
3. `python -m traceguard.cli doctor` — registry + split-manifest + template-attack validation OK.
4. `python -m traceguard.cli smoke --out runs/smoke-<shortsha>` — full pipeline smoke OK.

**Per run (every generation/monitoring batch):**
5. `python -m traceguard.cli verify --run-dir <run>` — schema validity, uniqueness, ledger completeness, outcome re-grade match, score-id uniqueness, checksums.

**Per gate (M4, M6, M7, M8, M9):**
6. Multi-POV review per `docs/REVIEW_SYSTEM.md` with recorded verdicts.
7. Precision/power check before any low-FPR claim (≥40 independent benign template clusters for a 5% FPR target; otherwise recall@FPR is exploratory and paired AUROC difference is primary per the preregistered rule).

**Per figure:**
8. Every figure embeds run id, config hash, and data digest; `.plotdata.json` is emitted next to each PNG.

## 4. Evidence and provenance rules

- Episode ids are opaque (never encode honest/attack); identifiers leaking labels fail `is_opaque_identifier`.
- Monitor views are allowlist-compiled; `assert_view_is_clean` runs before every scoring call and in tests for every condition.
- `hidden_instruction` events are excluded from ALL views (their mere presence would label attacked episodes).
- Run directories are frozen to one config hash; drift aborts execution.
- Storage is append-only, atomic (temp→fsync→rename), and resumable without duplication.
- Every run ships: run manifest, plan manifest, environment probe (GPU), episode shards, privileged audit shards, score shards, checksums, logs.

## 5. Statistical integrity

- One primary comparison (C vs A at calibrated FPR, or preregistered fallback to paired AUROC difference), frozen before holdout.
- Clustering respected: template-level cluster bootstrap for CIs; no pseudo-replication of near-identical instances.
- Thresholds calibrated only on development/validation splits; realized FPR on evaluation splits is always reported.
- All exclusions, refusals, timeouts, and failures appear in censor logs with counts.
- Exploratory analyses are labeled exploratory and may never be promoted to confirmatory without a new holdout.

## 6. Safety constraints (hard scope limits)

- The sandbox is in-memory, typed, network-free, shell-free. No real credentials, hosts, or production systems.
- Model-generated text is data; it is never executed. Tool calls are strict JSON validated against typed schemas.
- No training against deployed safety systems; no frontier-model claims beyond the documented threat model boundary.
- Raw attack traces and hidden instructions stay out of monitor views and are release-reviewed before any publication.

## 7. Failure handling ladder

1. **Stop** on contradiction of this manual.
2. **Record** the failure in the worklog and, if it changes frozen parameters, in `docs/deviations.md`.
3. **Repair** the methodology (new benchmark version + new preregistration if needed) — never the data.
4. **Re-run** the verification cadence from step 1 of §3.
