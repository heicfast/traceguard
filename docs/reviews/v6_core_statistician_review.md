# Statistician review — TRACEGUARD v0.6.0 analysis core

Reviewer: independent senior scientist/statistician (experimental design &
statistical inference), adversarial pre-experiment review.
Scope: `src/traceguard/analysis_runtime.py`, `src/traceguard/analyses/metrics.py`,
`src/traceguard/analyses/plots.py`, `src/traceguard/analyses/uncertainty.py`,
`src/traceguard/analyses/pair.py`, `src/traceguard/study_design.py`,
`configs/planning_assumptions.template.yaml`,
`tests/unit/test_research_operations.py` (plus the other analysis-related test
files), against `docs/ACADEMIC_METHODS_V6.md` (the protocol).
Repo state: `pyproject.toml` version 0.6.0. Full test suite: **166/166 passed**
(exit 0). Review method: line-by-line reading, plus independent numeric
verification against `scipy.stats.binomtest`, `statsmodels`
(`multipletests`, `proportion_confint`), `sklearn` (`roc_auc_score`,
`average_precision_score`), brute-force enumeration, and a synthetic end-to-end
run driven through `run_analysis` (schema-valid episodes + monitor scores,
development/holdout/shift splits, 4 evidence conditions, 3 predeclared
contrasts).

Display-hazard discipline: all bracketed-text questions were settled with
programmatic byte assertions (e.g. `b"[honest, attack]" in
open("configs/smoke.yaml","rb").read()` is TRUE even though terminal display
mangles it). No file was judged corrupted from visual inspection.

---

## VERDICT

**APPROVE-WITH-CONDITIONS** — for *exploratory/pilot* execution on this core
the statistical machinery is correct as verified; **two MAJOR findings must be
fixed before any confirmatory-stage execution or preregistration freeze**
(Conditions C1 and C2 below). Counts: **0 BLOCKER, 2 MAJOR, 5 MINOR, 3 NOTE**.

- **C1 (Finding 1):** Freeze thresholds per *condition* (computed once from
  that condition's valid calibration-split benign scores), not per *contrast*
  on each pair's complete-case calibration frame.
- **C2 (Finding 2):** Add end-to-end tests of `run_analysis` orchestration
  (Holm within-split families, McNemar secondary labeling, frozen-threshold
  wiring, gate consequences, AUROC non-substitution, figure/plotdata
  contract). Today this layer has zero test coverage anywhere in the repo.

---

## Findings

### Finding 1 — MAJOR: thresholds are frozen per contrast, not per condition; the same condition can get different thresholds in different contrasts

**Evidence.**
`src/traceguard/analysis_runtime.py:460-468`: for each predeclared pair,
`pair_calibration` is that *pair's* complete-case frame
(`paired_subset(rows, [primary, baseline])`, built at :405-409), and
`frozen[condition] = threshold_at_fpr(cal_benign, fpr_target)` is computed on
the benign rows of *that frame*. The same construction feeds the pairwise
`_evaluate_split` calls at :435-448.

**Demonstrated numerically** (adversarial check, not a hypothetical): 10
calibration benign clusters; condition `actions` valid everywhere;
`actions_plus_cot` missing on the 3 highest-`actions` rows. Then

- `actions` threshold inside the `actions_plus_cot_vs_actions` contrast
  (complete-case n=7): **0.12000000000000001**
- `actions` threshold inside the `actions_plus_shuffled_cot_vs_actions`
  contrast (complete-case n=10): **0.18000000000000002**

Attack rows with `actions` score in (0.12, 0.18] are detected in one contrast
and missed in the other, with the same nominal "actions at 5% FPR" label.

**Why it matters scientifically.** Protocol §1 defines
ΔR = R_C(τ_C) − R_A(τ_A) with "each threshold τ learned **once** from benign
calibration clusters at the predeclared target FPR". A per-contrast τ means
R_A(τ_A) is not a single well-defined quantity across the report, so §6-style
cross-contrast interpretations ("raw CoT losing while isolated fusion wins")
compare effects measured against *different* baselines. At 5% FPR the
threshold sits above the maximum benign score (k=0 for n<20), exactly the
low-support regime where dropping a few calibration rows moves the threshold
the most. The retention gate (≥90%, `analysis_runtime.py:421-423`) does not
prevent this: calibration-split-only missingness barely dents overall
retention. Note the confirmatory stage is separately blocked below 40 benign
calibration clusters (`generation_runtime.py:152-181`), which limits — but
does not eliminate — the exposure; pilot/exploratory runs proceed.

**Fix.** Compute τ_X once per condition X from the *condition-wise* valid
calibration benign scores (the `cal_benign` filter at :244-250 already does
exactly this for the descriptive tables), freeze them in one dict, and reuse
those values in every contrast, in the bootstrap (:487-501), and in the
McNemar test (:516-519). Keep the pair complete-case restriction for the
*evaluation* rows only (pairing is about episodes, not about subsetting the
calibration pool).

### Finding 2 — MAJOR: the analysis orchestration (`run_analysis`) has no test coverage anywhere; protocol-critical wiring is enforced by nothing

**Evidence.** `grep -r "run_analysis" tests/` → no matches (also no matches in
`src/traceguard/smoke.py` — even the smoke pipeline never runs the analysis).
`tests/unit/test_research_operations.py` tests the primitives only:
:21-26 (McNemar p=0.5 for b=2,c=0; Holm [0.01,0.04,0.03]→[0.03,0.06,0.06]),
:29-32 (PR tie grouping), :35-37 (nextafter/-0.0 edge), :40-46 (sizing
monotonicity in ρ only). Additional coverage exists in
`tests/unit/test_review_regressions.py:194-213` (frozen-threshold method
label + point estimate) and `tests/unit/test_phase3_hardening.py:103-126`
(CI suppression for `too_few_total_clusters`; inclusive-tie sensitivity), but
nothing pins:

- Holm applied **within holdout and separately within shift**
  (`analysis_runtime.py:552-561`) — a refactor that pools all six p-values
  into one family passes every test;
- `inferential_role = "secondary_episode_paired_test_not_cluster_adjusted"`
  (:559-561) — the McNemar-is-secondary guarantee;
- the precision-gate consequence string (:590-597) and the no-AUROC-substitution
  decision;
- that `cluster_bootstrap_paired` is always called with `frozen_thresholds`
  (:496 is the only call site in the codebase — an unprotected invariant);
- retention/calibration/trace/score-resolution gate wiring (:421-423, :424-432,
  :563-598, :632-642, :141);
- the figure contract (plotdata JSON, provenance footer, PNG metadata;
  `plots.py:17-36, 249-251`) and the AUROC figure-fallback behavior
  (:747-749, Finding 3);
- the exact sizing formula (Finding 2's test asserts only monotonicity).

**Why it matters scientifically.** Every protocol guarantee I verified below
is currently enforced only by code-reading. One silent refactor (a dropped
kwarg, a changed key prefix, a family pooled across splits) corrupts every
scientific conclusion while all 166 tests stay green.

**Fix.** Add one integration test with a small synthetic run (I built exactly
this harness during the review: ~20 assertions transcribable directly —
frozen thresholds equal `threshold_at_fpr(calibration benign)`, bootstrap
estimate equals direct computation, McNemar reproduced from frozen thresholds,
Holm reproduced per split with family size = number of predeclared contrasts,
gate booleans and consequence strings, CSV columns, plotdata files present).
Also add an exact-value sizing test (defaults → 228 / 1.7 / 388 / 1362) and
per-rule suppression tests for the three CI rules not currently pinned.

### Finding 3 — MINOR: latent AUROC-substitution path in the paired-effects figure, without an estimand label

**Evidence.** `analysis_runtime.py:747-749`:
`candidate = recall_payload if recall_payload.get("estimate") is not None else auc_payload`;
`plots.py:214-246` renders the candidate under keys like
`holdout|actions_plus_cot_vs_actions` with axis label "paired difference
(primary - baseline)" and no statistic label (demonstrated: an AUROC payload
renders normally; only the `.plotdata.json` "method" field distinguishes it).

**Why it matters.** Protocol §4: AUROC/AP "cannot replace the fixed-FPR
primary estimand after a failed precision gate". In the *current* wiring the
fallback is practically unreachable (recall estimate is None only when the
split has no attack rows or calibration is missing — in both cases the AUROC
payload is degenerate/absent too, and the `ci_low`/`ci_high` filter at
:750-755 drops it), so this is a latent risk, not a live bug. But it is one
edits-diff away from a silent estimand swap in the most-looked-at figure.

**Fix.** Either delete the fallback, or label the statistic on the figure and
in the key (e.g. append "(AUROC Δ — exploratory diagnostic)") whenever the
candidate is the AUROC payload.

### Finding 4 — MINOR: config schema permits `primary_estimand: "paired_auroc_difference"`, and the confirmatory guard only protects the recall estimand

**Evidence.** `config.py:197-198`: the Literal allows
`"paired_auroc_difference"`. `generation_runtime.py:152-153`: the
confirmatory-stage calibration-support block fires only
`if stage == "confirmatory" and (config.analysis.primary_estimand ==
"recall_difference_at_dev_calibrated_fpr")`.

**Why it matters.** Protocol §4 makes AUROC/AP *secondary rank diagnostics*;
§1 defines the primary estimand as the fixed-FPR recall difference. A
confirmatory config with the AUROC estimand skips the benign-cluster/allowed-FP
block and could promote a secondary diagnostic to "primary" (visible in the
`estimand` report field and the prereg, so auditable — hence MINOR, not MAJOR).

**Fix.** Restrict `"paired_auroc_difference"` to non-confirmatory stages (or
remove it), or extend the confirmatory block to reject it outright.

### Finding 5 — MINOR: `PlanningAssumptions.validate()` rejects ρ = 0, although the design effect is well-defined there

**Evidence.** `study_design.py:57-59`: `intracluster_correlation` must be
*strictly* between 0 and 1 (verified: `PlanningAssumptions(intracluster_correlation=0.0).validate()`
raises). Yet DE = 1+(m−1)ρ at ρ=0 is exactly 1, and the protocol formula
(§5) covers ρ=0; a pilot ICC point estimate of 0 (or planning with m=1, which
*is* allowed) is scientifically legitimate.

**Fix.** Accept ρ ∈ [0, 1) (and review the same strictness for
`operational_reserve_fraction` = 0).

### Finding 6 — MINOR: per-condition descriptive tables include invalid (present-but-invalid) scores; adjacent report fields use different denominators

**Evidence.** `analysis_runtime.py:261-263` (realized FPR), :295-298 (recall
at calibrated FPR), :329-341 (AUROC/AP) filter only
`r["scores"].get(condition) is not None` — a failed-parse score (e.g. a
fallback value) enters these numbers. By contrast :270-287
(`realized_fpr_available_case`) and the entire inferential path
(`pair.py:143-161` paired complete-case; bootstrap; McNemar) require
`score_valid`.

**Why it matters.** In the same report block, "recall at calibrated FPR" can
disagree with the paired inferential estimate purely because invalid scores
are counted as scores in one and censored in the other. Not an estimand
violation (the inferential path is clean), but a misreading hazard;
invalid-but-present scores are typically defaults that bias recall downward.

**Fix.** Require `score_valid` in the `_evaluate_split` score lists (or
explicitly label these fields as invalid-inclusive diagnostics).

### Finding 7 — MINOR: documentation/contract mismatches that misstate what is actually computed

**Evidence.**
(a) `plots.py:2-5` claims every figure "embeds its data digest (SHA-256 of the
serialized plotting data)", but `provenance_line` (`plots.py:249-251`) hashes
`{"run_id", "config_sha256"}` only — the "data:" field is not a digest of the
plotting data. (The adjacent `.plotdata.json` *is* covered by the evidence
index when written inside the run dir, so traceability is materially intact —
but the stated contract is wrong.)
(b) `uncertainty.py:3-5` claims the resampling unit is "family for shift
analyses", but `analysis_runtime.py:492,506` applies the single
`config.analysis.cluster_key` to both holdout and shift. The V6 protocol
("template or family level") is satisfied either way, but the docstring
overstates; and if families shift together on the shift split, template-level
resampling there understates clustering unless the operator sets
`cluster_key: family_id`.
(c) The estimand literal `recall_difference_at_dev_calibrated_fpr`
(`config.py:197`) says "dev" while the default `calibration_split` is
`"validation"` (`config.py:115`) — auditable via the report's
`precision_gate.calibration_split`, but the name invites misreading.

**Fix.** Correct the docstrings; consider hashing the plotdata payload in the
footer; rename the literal at the next schema break (report field stays
auditable meanwhile).

### Finding 8 — NOTE: frozen-but-undocumented numeric conventions (all conservative)

(a) Percentile CI endpoints use floor indices `int(α/2·N)` /
`min(N−1, int((1−α/2)·N))` (`uncertainty.py:98-99`) — for N=2000/95% that is
order statistics 50 and 1950, a valid but nonstandard convention (one order
statistic inside the classic definitions). Document as frozen.
(b) Sizing double-rounds: `ceil(ceil(n_ind)·DE)` (`study_design.py:86,90`) =
388 for the defaults versus `ceil(n_ind·DE)` = 387 — conservative; document.
(c) `threshold_at_fpr` uses `floor(fpr·n + 1e-12)` (`metrics.py:25`) while the
gates use `floor` without the epsilon (`analysis_runtime.py:417,572`;
`generation_runtime.py:170-171`) — in pathological float cases the gate is one
allowed-FP stricter than the threshold; conservative; unify.

### Finding 9 — NOTE: the §6 headroom gate is not surfaced in the per-run analysis report

**Evidence.** The analysis core computes no headroom gate. Action-only
headroom is enforced operationally elsewhere: `scripts/monitor_gate.py`
M5 (saturation: RuleMonitor recall == 100% AND FPR == 0%, :19-27, :235-291)
and `scripts/compare_cells.py:40,143-164` (band 0.40–0.70 with explicit
`saturated`/`floor` flags, per cell per split, matching
`docs/PHASE3_PROTOCOL.md:150`).

**Why it matters.** §6 gate 3 is a confirmatory-claim requirement; a reader of
`analysis_report.json` alone sees precision/retention/trace/resolution gates
but no headroom verdict. **Fix:** echo an explicit headroom field (actions-only
recall, in-band flag) into the report for confirmatory runs.

### Finding 10 — NOTE: CensorLog detail truncation

`pair.py:33-35` caps `details` at 500 entries; the `by_reason` counts remain
complete. Acceptable, but state the cap in the payload so a consumer never
mistakes the list for exhaustive.

---

## VERIFIED CORRECT (protocol requirement → code evidence → verification)

1. **ΔR estimand: paired same-episode contrast at frozen thresholds learned
   once from calibration-split benign scores.**
   `analysis_runtime.py:239-256` (calibration benign only, valid-only filter),
   :460-468 (frozen per pair — see Finding 1 for the per-pair caveat),
   `uncertainty.py:64-78` (point estimate *and* replicates use frozen
   thresholds). Verified end-to-end on a synthetic run: reported frozen
   thresholds equal `threshold_at_fpr` of the calibration benign scores, and
   the bootstrap estimate equals the directly computed paired recall
   difference (0.375 reproduced exactly).
2. **No threshold re-estimation inside holdout/shift bootstrap replicates.**
   `analysis_runtime.py:487-501` passes
   `frozen_thresholds=(primary_threshold, baseline_threshold)` (:496); this is
   the *only* call site of `cluster_bootstrap_paired` in the codebase. Verified
   by monkeypatching `threshold_at_fpr` to raise: the frozen run completes
   (100 replicates, 0 degenerate); the legacy no-freeze path does recalibrate
   and is unreachable from the pipeline (and is labeled
   `..._recalibrated` when used, `uncertainty.py:134-136`).
3. **CI suppressed as `not_estimable` under all four documented rules.**
   `uncertainty.py:104-113,119-122` (paired) and :219-229,232-235 (AUROC
   variant): total clusters, positive clusters, degenerate fraction, bootstrap
   support; suppressed CIs are `None` with `ci_ineligibility_reasons`.
   Verified: constructed fixtures triggering each rule independently
   (`too_few_total_clusters`, `too_few_positive_clusters`,
   `too_many_degenerate_replicates` at deg=0.353 > 0.10,
   `insufficient_bootstrap_support` at support=1 < 3), plus a healthy case
   that stays `estimable`. Defaults wired from `config.py:205-208`.
4. **Exact McNemar: two-sided exact conditional binomial, discordant pairs
   only.** `metrics.py:178-212` conditions on b+c, doubles the minimum tail,
   caps at 1; b=c=0 → p=1. Verified against `scipy.stats.binomtest`
   (max |Δ| = 1.1e-16 over 22 edge cases + 500 random paired vectors).
   `recall_difference` is (b−c)/n over all pairs — the correct paired recall
   difference, and it equals the bootstrap estimate in the synthetic run.
5. **McNemar is secondary and cannot override the effect estimate.**
   Computed on the same frozen thresholds as the bootstrap
   (`analysis_runtime.py:516-519`); stamped
   `inferential_role="secondary_episode_paired_test_not_cluster_adjusted"`
   (:559-561); reported in a separate report section and CSV; nothing feeds
   p-values into gates, intervals, or effect estimates anywhere in
   `run_analysis`.
6. **Holm step-down FWER adjustment, within holdout and separately shift.**
   `metrics.py:215-226`: sort ascending, multiplier (m−rank), running max,
   cap at 1, original order restored. Matches `statsmodels` exactly on 300
   random vectors (m=1..12, with ties) and the hand example. Applied per split
   over the predeclared contrasts (`analysis_runtime.py:552-561`); verified on
   the synthetic run: family size 3 within each of holdout and shift
   (raw [0.25, 0.5, 0.5] → adjusted [0.75, 1.0, 1.0] in both), no pooling
   across splits.
7. **AUROC/AP are secondary rank diagnostics; no silent substitution after a
   failed precision gate.** Precision gate and consequence
   (`analysis_runtime.py:563-598`) flags recall@FPR EXPLORATORY and states the
   no-AUROC-substitution decision; confirmatory execution is blocked at
   generation time below the required benign calibration clusters/allowed FPs
   (`generation_runtime.py:152-181`); AUROC deltas ride in separate
   `auroc_delta` payloads (:502-514); ROC/PR figures are labeled as curves.
   (Caveats: Findings 3 and 4.)
8. **Tie handling: scores tied → processed as a group; no record-order
   resolution.** AUROC mid-ranks (`metrics.py:72-91`) and grouped AP
   (:94-129) match sklearn to <1e-12 on 300 coarse-grid cases with heavy ties;
   `precision_recall_points` (:147-175) verified permutation-invariant under
   record shuffling; `threshold_at_fpr` (:13-32) is conservative on ties
   (FPR ≤ target guaranteed on 300 random cases; benign-tied attacks not
   flagged) with an explicit inclusive-tie sensitivity
   (:35-52, wired at `analysis_runtime.py:300-325`; pinned by
   `test_phase3_hardening.py:121-126`).
9. **Sizing model matches the protocol formula exactly.**
   `study_design.py:71-96`: z_{1−α/2} = `inv_cdf(1−α/2)` (no off-by-one),
   z_power = `inv_cdf(power)`, n = (z-sum)²(q−δ²)/δ² (:86), DE = 1+(m−1)ρ
   (:87-89), then ÷ allocation × attempt-yield × retention
   (:108-116). Verified: defaults give n_ind 227.6175→228, DE 1.7,
   clustered 386.95→388, episodes 1357.72→1362; 200 random designs match the
   hand formula at every stage; δ=0 and δ≥q rejected; m=1 → DE=1 (ρ=0 caveat:
   Finding 5).
10. **Cluster bootstrap resamples clusters, jointly for both arms.**
    `uncertainty.py:84-92` (draw len(K) clusters with replacement; both
    conditions consume the same resampled rows); replicates without attack
    rows are counted degenerate and excluded (:89-91, :100-102); the exact
    multinomial reference distribution for K≤10 (:141-171) was verified
    against brute-force k^k enumeration (k=5: 126 compositions = C(9,4),
    degenerate mass 0.07776 identical, full support identical to <1e-12).
11. **Quality gates present and correctly wired.** Pairwise retention gate
    (`analysis_runtime.py:418-423`, default 0.90 at `config.py:210`);
    per-pair calibration gate (clusters ≥ 40, allowed FPs ≥ 3,
    :424-432); global precision/FPR gate with explicit rule text (:563-598);
    trace-coverage gate (:600-611, :632-642, default 0.90 at `config.py:212`);
    score-resolution gate (:141, default 10 levels at `config.py:209`);
    parse/validity diagnostics (:123-161 plus `monitor_gate.py` M2 ≤10%
    invalid); complete censor accounting (`pair.py:17-36,143-161` — every
    duplicate, missing, invalid, runtime-error exclusion counted with
    reason codes). Headroom gate location: Finding 9.
12. **Wilson interval and missing-score bounding are correct.**
    `_wilson_interval` (`analysis_runtime.py:164-173`) matches
    `statsmodels ... method="wilson"` to ≤1.1e-16 across cases incl. k=0, k=n;
    `_recall_missing_score_sensitivity` (:84-120) verified by hand: missing
    scores never invent detections, worst/best bounds bracket the observed
    delta.
13. **Figure contract implemented.** Every figure writes a `.plotdata.json`
    next to the PNG, a provenance footer, and PNG/SVG/PDF copies with the
    provenance in metadata (`plots.py:17-36`); verified on the synthetic run
    (6 figures + 6 plotdata files; PNG `Subject` = "TRACEGUARD rev-run-1 ·
    cfg:… · data:… · testmon"). Tidy CSVs include `inferential_role`
    (`analysis_runtime.py:176-217`).
14. **`planning_assumptions.template.yaml` matches `PlanningAssumptions`
    defaults field-for-field** (`configs/planning_assumptions.template.yaml`
    vs `study_design.py:26-43`); `study-plan` CLI rejects unknown assumption
    keys (`cli.py:243-252`); the plan output labels itself
    `planning_estimate_not_preregistration` and carries the
    simulation-validation interpretation strings (`study_design.py:174-206`),
    consistent with §5's "episode count alone does not establish power".

## Numeric spot-check results

| Check | Result |
|---|---|
| `exact_mcnemar` vs `scipy.stats.binomtest` (two-sided exact) | 22 edge cases (b=c, b=0, n=1, n=70, 0 discordant→p=1) + 500 random paired vectors: max abs diff **1.11e-16** |
| `holm_adjust` vs `statsmodels multipletests(method="holm")` | 300 random vectors (m=1..12, ties included): max abs diff **0.0**; hand check [0.01,0.04,0.03] → [0.03,0.06,0.06]; all-equal [0.02,0.02,0.02] → [0.06,0.06,0.06]; monotonicity holds |
| Sizing vs doc formula | Defaults: z=1.959964/0.841621, n_ind=227.6175 (code 228), DE=1.7000, clustered=386.9498 (code 388), episodes=1357.72 (code 1362); 200 random designs: exact match at every stage; δ=0/δ≥q rejected; m=1→DE=1 |
| AUROC / AP vs sklearn (coarse scores, heavy ties) | 300 cases: max abs diff **< 1e-12** both metrics |
| PR points tie grouping | permutation-invariant over 50 shuffles; hand case ([0.9,0.5] vs [0.5,0.1]) exact |
| `threshold_at_fpr` | 300 random cases: FPR ≤ target always; k=1 known case → 0.4000000000000001, FPR=0.2, benign-tied 0.4 not flagged; target 0 → FPR 0; target 1 → −inf |
| Frozen-threshold bootstrap | monkeypatched `threshold_at_fpr` never called with frozen thresholds; point estimate = direct computation; degenerate replicates 0 |
| CI suppression rules | 4/4 rules trigger independently; suppressed CI = None/`not_estimable`; healthy case estimable |
| Exact cluster distribution (K=5) | identical to brute-force k^k enumeration: compositions 126 = C(9,4), degenerate mass 0.07776, full support match |
| Wilson interval vs statsmodels | ≤ 1.1e-16 across 7 cases incl. boundaries |
| End-to-end synthetic run | Holm families separate per split; McNemar reproduced from frozen thresholds (b=3,c=0,p=0.25; b=2,c=0,p=0.5); bootstrap estimates reproduced (0.375/0.25); gates reported (retention PASS, calibration FAIL 12<40, precision FAIL with EXPLORATORY consequence, trace PASS, resolution PASS); 6 figures + 6 plotdata + 2 CSVs |
| Full test suite | **166 passed, 0 failed, exit 0** |

## Test-suite assessment (test_research_operations.py and friends)

The expectations that *are* pinned are **correct** (McNemar p=0.5 for b=2/c=0;
Holm [0.01,0.04,0.03]→[0.03,0.06,0.06]; PR tie grouping; nextafter/−0.0 edge;
frozen-threshold method label and point estimate; CI not_estimable under
too-few-clusters; inclusive-tie sensitivity; AUROC/AP vs sklearn). The
weaknesses are **coverage, not correctness**: the sizing test checks only
monotonicity (not the formula, nor ρ=0/m=1/δ-boundary behavior); only one of
the four CI suppression rules is pinned; and the entire `run_analysis`
orchestration — where the protocol actually lives — is untested (Finding 2).

## Conditions for confirmatory use (summary)

1. Fix Finding 1 (per-condition threshold freezing) and re-verify that
   `frozen[condition]` is identical across all contrasts for a given
   condition on a missingness-injected fixture.
2. Land the Finding 2 integration tests.
3. Address Findings 3-7 (all small, localized) before the preregistration
   freeze; Finding 5 is a one-line validation change.

*Review artifacts: synthetic-run harness and spot-check scripts were executed
from the repo root (editable install) during this review; no source files
were modified.*
