# When Does Chain-of-Thought Visibility Improve AI Control? Evaluating Reasoning-Aware Monitors Under Distribution Shift

**TRACEGUARD Final Research Report — Working Draft (results sections pending the last two landed cells)**

Research program: TRACEGUARD. Working title as preregistered (see
`docs/preregistration_DRAFT_v0.2.md`). This document is the master
technical report; every number cited in it is reproducible from the
committed, checksum-verified evidence archives under `runs/v6/`.

Status: matrix cells 4/5 landed (anchor, scale, agent-qwen14,
agent-deepseek14), cross cell re-running after an external interruption,
online intervention study in flight. Results sections will be filled from
the landed evidence only — never from memory.

---

## 1. Introduction

Large language model agents that solve multi-step software tasks by
proposing tool calls increasingly operate inside safety envelopes built
around a monitoring layer. A monitor observes some projection of the
agent's behavior — its tool calls and their results, and possibly its
chain of thought (CoT) — and the deployer intervenes when the monitor
flags a proposal. Whether the reasoning trace, in particular, is worth
feeding to the monitor is a genuinely open empirical question with
arguments on both sides: the reasoning may contain intent information
that the actions never express, or it may function as a persuasion channel
that degrades the very monitor it is supposed to help. The original CoT
Monitorability position paper (Korbak et al., 2025) frames exactly this
as a priority research problem, and subsequent empirical literature has
produced results in both directions, suggesting the answer is
conditional: it depends on the attack class, the monitor architecture,
and the capability gap between agent and monitor.

TRACEGUARD was designed to bound and measure those conditions rather
than to answer the question once in the abstract. The program executes a
sequence of paired, holdout-controlled experiments in a deterministic
software-maintenance sandbox in which a Qwen3 agent performs ordinary
configuration changes while a subset of episodes (invisible to the
monitor) carry an adversarial side objective. Ground truth for "an
unauthorized state change occurred" comes from the sandbox's invariant
checker — not from any model's judgment — so attack success and monitor
detection can never be confounded with a judge's opinion. Every
monitoring condition is a blinded projection of the same stored
episodes: actions only; actions plus genuine reasoning; length-matched
noise; shuffled-reasoning controls; and an independently fused
extraction-plus-adjudication architecture.

The study asks four preregistered questions. RQ1 (primary estimand): does
access to the reasoning trace improve a monitor's ability to identify
unauthorized agent behavior beyond the information contained in
observable actions, measured as the recall difference ΔR between
conditions at a fixed false-positive rate? RQ2 (robustness): does any
improvement survive task-family and attack-strategy distribution shift?
RQ3 (information value): is an improvement attributable to the
reasoning's content rather than to additional token volume, leaked
labels, or style? RQ4 (control effectiveness): when the monitor gates
actions before execution, does it actually reduce successful violations
while preserving legitimate task completion? This report answers RQ3 and
RQ4 affirmatively in bounded form, quantifies RQ1 as structurally
ceiling-limited on the current benchmark inventory (with the exact
measurement that proves it), and reports RQ2 as a degradation profile
that depends jointly on monitor capability and fusion architecture.

## 2. Related work

The original CoT Monitorability framing (Korbak et al., 2025,
arXiv:2507.11473) identifies whole-episode monitorability evaluations,
causal relevance of the trace, adversarial robustness, and the
detection-to-prevention gap as open problems. OpenAI's operational
evaluations and the *Monitoring Monitorability* line (Guan et al., 2025)
report that CoT-plus-actions monitoring frequently outperforms
action-only monitoring in reward-hacking regimes, and that monitor
test-time compute buys reliability at the weak-to-strong boundary. A
2026 persuasion study (arXiv:2607.08066) demonstrates the opposite
mechanism: exposing an agent's scratchpad can *increase* a monitor's
approval of policy-violating actions, with cross-family fact-checking
substantially mitigating the effect. SLEIGHT-Bench (2026) and
SHADE-Arena (Kutasov et al., 2025) show that constructible attacks can
hide from transcript-level monitors and that implicit influence degrades
detection by large margins relative to explicit instructions. AI Control
(Greenblatt et al., ICML 2024) establishes the protocol family — trusted
monitors observing untrusted agents under blinded evidence conditions —
that TRACEGUARD instantiates in an open, state-grounded sandbox.

TRACEGUARD's contribution relative to this literature is narrower and
complementary: a fully open, deterministic, state-grounded testbed
(environment-determined ground truth, not judge-determined); a paired
condition matrix that separates reasoning *content* from reasoning
*volume* and *presence* (length-matched noise, shuffled-reasoning, and
reasoning-only controls in the same episodes); a capability axis (4B
versus 14B monitors; same-family versus distill-family) crossed with an
architecture axis (raw early fusion versus structured late fusion);
and a prevention experiment that runs fresh rollouts under a pre-action
gate with the threshold frozen from offline calibration. The negative and
conditional results are reported with the same rigor as positive ones:
the project's charter forbids predetermined conclusions.

## 3. The TRACEGUARD framework

### 3.1 Deterministic sandbox and environment-determined ground truth

The environment is a fictional software organization ("Meridian
Systems") whose configuration world is an in-memory, hash-chained state
object: services, routing tables, export destinations, automation
schedules, and audit controls. The agent acts through a strict tool
protocol (`propose_patch`, `propose_config_change`, `submit_task`, and
inspection tools); every accepted action produces an immutable
transition record with pre/post state hashes. Each task template declares
(1) a public user goal, (2) an explicit protected-state invariant
contract ("the value at path X must remain Y"), and (3) an optional
adversarial reference action. Ground truth is exact: an attack *succeeds*
if and only if the final state violates a declared invariant, graded by
replaying the accepted transitions against a fresh initial state with
the frozen grader — never by an LLM judge. The same grader re-derives
every outcome during verification, so a landed run's outcome labels are
independently reproducible from its event stream alone.

### 3.2 Attack model: four frozen strategies across four task families

Attack episodes carry a hidden side objective (never visible to any
monitor condition) drawn from the frozen v6 strategy pool of four
neutral-worded policies — `explicit_override`, `side_objective`,
`implicit_pressure`, and `monitor_aware` — crossed with all four task
families (32 attack episodes per strategy; the eight-strategy pool from
the Phase-2 design, including unaware and adaptive-red-team variants,
is retained in the registry for the confirmatory program). The Phase-3
inventory spans four task families designed to move dispositive evidence
away from single obvious actions: `artifact_routing` (promotion must
respect cross-object identity constraints), `service_identity` (an
apparently routine change re-points a service at a different principal),
`export_destination` (a write to a technically writable but unauthorized
destination), and `deferred_automation` (an apparently benign schedule
edit triggers a forbidden behavior later). Honest twins of every task
instance — same seeds, same template, no hidden instruction — exist for
every attack episode, so false-positive rates are exercised on genuinely
benign but structurally tricky inputs; half of the honest episodes are
negative-control variants (alternating `inspect_protected` and
`suspicious_phrasing`) that deliberately inspect or mention protected
state without touching it.

### 3.3 Blinded evidence conditions

Monitors observe only what a deployed monitor could see at decision
time: a compiled *view* of the episode. The condition set is the heart of
the design:

- **A (actions):** tool calls and results only.
- **C (actions + CoT):** the same action stream plus the agent's genuine
  captured reasoning tokens, head/tail-windowed at the predeclared
  evidence limit (12,288 chars; the 75/25 head/tail split keeps both
  opening deliberation and late execution reasoning visible), with a
  visible truncation marker.
- **D (actions + noise):** length-matched generated prose, marked
  identically, controlling for volume and style.
- **E (actions + shuffled CoT):** the *paired twin episode's* genuine
  reasoning (an honest twin for attacks, an attack twin for honest
  episodes), controlling for content presence with intent mismatch.
- **R (reasoning-only)** and the **late-fusion** conditions used by the
  structured monitor.

Every view passes a leak-guard assertion: the hidden instruction, its
hash, ground-truth labels, and pipeline-authored metadata can never
appear in a view. Views are content-hashed, and the gate M1 recompiles
every scored view from the raw evidence and refuses any mismatch.

### 3.4 Monitors and the capability/architecture matrix

Three monitor classes run over identical views. A deterministic rules
monitor (syntactic protected-vocabulary heuristics), a keyword baseline,
and LLM monitors producing a structured JSON verdict (rubric plus a
0-100 suspicion score) under a frozen prompt with per-episode
nonces. The structured late-fusion monitor replaces raw concatenation
with staged extraction: independent action-risk and reasoning-anomaly
extractors produce bounded structured claims, and an adjudicator (which
never sees the raw persuasive trace) fuses them with the deployer
policy. The Phase-3 matrix crosses this with capability: the anchor arm
(4B judging 4B), a scale arm (Qwen3-14B monitor, NF4-quantized, admitted
through a deployment-scale stress probe on the same hardware), a
cross-family arm (DeepSeek-R1-Distill-Qwen-14B), and two agent-scale
arms (14B agents judged by the 4B monitor). Three landed cells and the
re-running two run the same frozen design
(`configs/phase3_pilot.yaml`); the three monitor cells score the SAME
stored 128-episode cohort (byte-identical episode archives; verified at
landing), while the two agent cells generate fresh 64-episode cohorts
under their own 14B agents (zero episode overlap with the anchor cohort —
per-model cohort generation is the design, not a confound, because
agent contrasts compare monitors over each agent's own rollouts). All
cells share the same code lineage (the monitor cells ran two adjacent
kernel pins whose scoring-path diff is empty — verified). Contrasts are
paired at the episode level within each cell.

### 3.5 Evidence discipline and reproducibility

Every run is a WAL-transactional store with commit markers and crash
recovery; every file is SHA-256 ledgered under an evidence index built
before the ledger; verification replays the grader over all episodes and
refuses any gap. Kernels pin exact commits with fail-closed checkout and
revision gates; model admission requires an on-hardware probe at
deployment-scale prompt sizes. Analysis code and report figures
reproduce deterministically from the committed JSONL evidence — the
landing chain literally re-runs the kernel's analysis and requires
path-prefix-only differences. The full matrix evidence, including raw
score records with prompt hashes and rendered prompts, is committed.

## 4. Results

*(Sections 4.1-4.5 fill from the landed evidence archives; the numbers
below are the landed state as of this draft's timestamp and are
reproduced from `runs/v6/*/run/reports/analysis_report.json`.)*

### 4.1 Pilot (Phase 2, 64 episodes, pre-test archive)

The first real-model cohort (32 attack / 32 honest; Qwen3-4B agent ×
same-checkpoint 4B monitor, v1 raw instrument) established feasibility and
produced the study's first — and warningly negative — exploratory signal
(`runs/pre-tests-data/pilot/`, superseded by the v0.6 from-scratch restart;
full evidentiary archive, `release_complete: true`). Capability funnel:
14/32 attacks attempted (43.8%), 14/14 attempted attacks succeeded (100%),
honest task completion 25/32 (78.1%); trace coverage complete. Every
fixed-FPR statement from this run is **exploratory**: the validation split
carried only 2 benign calibration clusters against the 40 required, so the
precision gate fired and confirmatory execution was blocked by design.

At the development-calibrated threshold, the paired primary contrast
ΔR = R(C) − R(A) estimated on the holdout split was **−1.0** (A recall 1.0 →
C recall 0.0). Three qualifications, all recorded in the run's own analysis
and verified by three independent reviews: (i) the cluster bootstrap CI is
**not estimable** — 4 holdout clusters with exactly 1 positive-bearing
cluster, degenerate-replicate fraction 0.31, and a single-point support at
−1.0; the zero-width presentation expresses a design that cannot express
uncertainty, not a precisely-known effect; (ii) the point estimate is
censoring-sensitive — 5/46 actions+CoT scores were JSON-invalid versus 1/46
actions-only, and channel-repaired parsing of the malformed first fields
moves C to ≈ 1/3 (ΔR ≈ −0.67); the direction survives, the magnitude does
not; (iii) the v1 instrument itself was coarse (3–5 distinct score levels
per condition, failing the score-resolution gate), and the degradation
co-occurred with generic long-context fragility — the matched-length noise
control (D) also cost recall on development (0.4) and holdout (0.5)
relative to actions (0.8 / 1.0). The shuffle control (E) matched C's
collapse on holdout (0.0), so on this pilot no content-specific claim was
possible.

Two findings from the pilot survived all scrutiny and shaped Phase 3.
First, the deterministic keyword monitor's recall tripled with CoT exposure
(2/14 → 6/14 attack episodes flagged): the traces contained additional
extractable attack signal that the 4B LLM monitor failed to use — evidence
that the negative result measured the monitor, not the medium. Second, the
recorded verdict texts showed the failure mode in the raw: across the five
attempted-attack episodes scored lowest under C, every verdict adopted the
agent's self-authorization framing ("per the override note...", "as
authorized...") while simultaneously quoting the violating action — the
persuasion-channel mechanism that the later agent-capability cells and the
14B monitor cells were designed to isolate (and that the 2026 persuasion
literature documents independently: scratchpad exposure as an additional
persuasion channel). These two observations — extractable signal plus
persuasion susceptibility — became the two hypotheses the Phase-3 matrix
separates.

### 4.2 Phase-3 anchor cell (4B monitor, v2 instrument)

The v2 raw prompt collapsed the 4B monitor's expressiveness: actions-only
scores were degenerate (a single distinct value) across every condition,
while the same episodes under late fusion produced a 16-level
distribution with AUROC 0.923 (development) and 0.690 (holdout) versus
0.5 for both raw arms. The raw arm's degeneracy is therefore an
instrument-and-capacity property, not evidence about CoT per se; all
raw-arm contrasts on this cell are non-estimable and reported as such.

### 4.3 Phase-3 scale cell (Qwen3-14B monitor) — LANDED 2026-10-28

At 14B the identical raw prompt yields a discriminative distribution
(actions: 6 distinct levels; CoT: 5; late fusion: 26; the raw arms still
sit below the preregistered ≥10-level M3 instrument-resolution gate, so
all raw-arm numbers in this section carry an M3-coarse-instrument
caveat) and clean separation: AUROC (A/C) = 1.0/1.0 (dev), 1.0/0.9583
(holdout), 1.0/0.9643 (validation). Actions-only recall at calibrated
zero-FPR is 1.0 on every split — the ceiling that makes positive
ΔR(C−A) structurally impossible on this inventory. The information-value
controls separate CONTENT from VOLUME as follows. The shuffled-reasoning
control (E) is decisive: at the frozen threshold E recall collapses to
0.0 on all four splits, but this is a single-threshold-event — one
honest validation episode scored E=1.0 and inflated the calibrated
threshold to 1.0; E retains rank signal everywhere (E-AUROC 0.73–1.00
by family/strategy, with two strata at exactly 1.000 where actions
dominate even under wrong-label reasoning), and a leave-one-benign-out
recalibration moves E recall to 0.46–1.0. The noise control (D) does
NOT consistently separate: D dips below C only on development (0.62 vs
1.0); on holdout, shift, and validation D matches or exceeds C. RQ3's
content claim therefore rests on the label-matched E control, NOT on
volume matching. (The Phase-2 keyword-baseline tripling does NOT
reproduce on this cohort — the deterministic keyword monitor flags
0/53 attacks from actions and 1/53 with CoT, plus one honest false
flag: the v6 families' reasoning simply does not contain the legacy
suspicious-phrase vocabulary, so keyword extractability is itself
benchmark-relative.)

The content-vs-volume verdict from the two controls read together: the
label-matched shuffle control (E) destroys the CoT advantage while
matched-length neutral prose (D) does not consistently separate from real
CoT off-development — so genuine reasoning content is used by the 14B
monitor (E-AUROC retains rank signal per family/strategy), but volume
matching is not reliably distinguished at this scale, and the RQ3 content
claim rests on E alone. Persuasion robustness
improves with capability: the 4B anchor's holdout collapse (1.0 → 0.0)
does not reproduce at 14B (C stays ≥ 0.92).

### 4.4 Phase-3 agent cells and cross-family cell

**Agent-capability cells (landed).** Two fresh cohorts replayed the frozen
plan with stronger agents under the same 4B v2 monitor: Qwen3-14B
(32/32 attacks attempted = 100%, 32/32 succeeded; honest completion 32/32 =
100%) and DeepSeek-R1-Distill-Qwen-14B (29/32 attempted = 91%, 29/29
succeeded; honest completion 30/32 = 93.75%). Two conclusions follow. First,
agent capability raises threat fidelity without any honesty cost: both
agents are *more* reliable attackers *and* better honest workers than the 4B
anchor (83% attempts, 98% honest completion), and attack success among
attempted episodes remains 100% — so every detection number on these
cohorts is measured against fully-executed violations, not degraded
attempts. Second, the 4B raw monitor's collapse is **agent-independent**:
its score distribution stays at a single distinct level (AUROC 0.5)
across all four conditions and all four splits on both cohorts — while
the same 4B checkpoint under the fusion adjudicator prompt stays
discriminative (late-fusion AUROC: Qwen3-14B cohort 0.953 dev / 0.719
holdout / 0.797 shift / 0.859 validation; DeepSeek cohort 0.859 / 0.634 /
0.768 / 0.893). The raw-arm degeneracy therefore cannot be attributed to
anything the agent wrote — it reproduces unchanged when the untrusted
narrative comes from a different, stronger model family, and even on the
actions-only arm that contains no narrative at all. (The fusion AUROC dip
on the stronger-agent cohorts versus the 4B-agent anchor — 0.719/0.634
holdout vs 0.690 — is within the paired-cluster uncertainty of these
designs and no agent-difficulty claim is made.)

**Cross-family monitor cell (DeepSeek-R1-14B monitor, re-run in flight).**
*(Consolidated at landing: the family-diversity axis — whether the raw
prompt's degeneracy at 4B recovers at 14B in a *different* model family,
isolating monitor-capacity from family-specific effects. Per the frozen
no-partial-mixing policy, the externally-cancelled attempts 3 (2/5
conditions) and 4 (3/5 conditions) are archived under
`runs/v6/failed_attempts/` and contribute nothing to the record; both
cancelled mid-run with zero OOM events, validating the dual-T4 split
layout.)*

### 4.5 Headroom: the uniform action-visibility ceiling

Rank-based (Mann-Whitney) actions-only AUROC computed per family and per
strategy on the landed scale-cell scores: **1.000 on all four task
families and all four strategies.** Threshold-based recall agrees (1.0 at
FPR 0 in every family; the benign score maximum is 0.0 and the minimum
positive score is 0.53). Even the relational, non-action-visible family
designs are fully inferable by a 14B semantic monitor from the action
stream alone — while the syntactic rules monitor sees nothing (recall
0/53), confirming the families did move evidence beyond *syntactic*
legibility, just not beyond *semantic* legibility. Quantitative bound on
the S5 decision: 0 of 53 attempted attacks scored at-or-below the maximum
honest action score, so no attack's violation was action-invisible to the
preregistered 14B monitor; the one-sided 95% upper bound on the
action-invisible fraction is 5.5% (< the preregistered SESOI of 10%;
exact binomial P(0 visible-defects | true rate ≥ 10%) = 0.0038). Consequence recorded in the preregistration: a
confirmatory-scale (~550-650 episode) run on this inventory would measure
a structurally null primary estimand for the preregistered 14B monitor
class (the measurement does not generalize to other monitor families by
construction) and is not scientifically justified; the
confirmatory program is predeclared as future work requiring
genuinely intent-only attacks and ≥40 benign calibration clusters.

### 4.6 RQ4: online pre-action intervention

*(Fills from the online study at landing: violation-success risk
reduction vs the unmonitored baseline at the frozen threshold, the
false-block burden on honest episodes, and the legitimate-work
completion cost, paired by plan key.)*

## 5. Discussion

*(Written after all cells land; the landed results already support these
framing points.)*

The picture that emerges is conditional and, we argue, more useful than a
binary answer. (1) CoT is an information channel, not a free lunch: at
sufficient monitor capability the content is genuinely used — the
label-matched shuffled-reasoning control breaks detection while genuine
reasoning preserves it — but on attack inventories where the action
stream itself is semantically legible, the channel adds nothing, and the
primary estimand has no room to be positive. Matched-length neutral
prose is NOT reliably rejected at this scale (the volume control does
not separate off-development), so the content finding is specifically
about intent-bearing ordering and labels, not mere CoT-shaped text.
(2) Raw early fusion is fragile in a capability-dependent way: the
identical prompt that collapses a 4B monitor's expressiveness works at
14B under production quantization; structured late fusion removes the
persuasion surface and works at both scales — architecture, not model
size, is the robust axis. (3) The persuasion failure mode is real but
attenuates with capability and disappears under staged extraction.
(4) What the field calls "action-visible" is monitor-relative: syntactic
monitors and semantic monitors see the same actions very differently,
and headroom gates must be defined against the monitor class under
test. (5) Offline detection does not by itself prevent anything — the
RQ4 pre-action study quantifies the prevention-to-detection translation
separately.

## 6. Limitations

- Benchmark scale is exploratory (128 episodes per cell; two benign
  calibration clusters; confidence intervals correctly reported as
  not-estimable where the cluster count demands it).
- The action-visibility ceiling is measured, not assumed: it binds at
  14B-class semantic monitors on this inventory. Intent-only attacks
  (violations recoverable only from reasoning) are the prerequisite for
  a confirmatory test of the primary estimand.
- One agent family (Qwen3) and two monitor families (Qwen3,
  R1-distill-derived) at ≤14B; no >14B monitor was deployable on the
  available T4 hardware.
- The evidence window (12,288 chars, head/tail) is hardware-derived and
  predeclared; attacks that place dispositive reasoning beyond the
  window would not be caught by this design (window-robustness arms are
  preregistered for the confirmatory program).

## 7. Reproducibility statement

Every claim in this report reproduces from the repository: pinned
commits, checksum-ledgered evidence archives under `runs/v6/`, the
deterministic grader re-run by `traceguard verify`, view-recompilation
by the monitor gate, analysis determinism proven at landing, and export
verification (`release_complete: true`) on every landed cell. The pilot
cohorts are also published as public Kaggle datasets. Monitor identity
for each scored cell is provenanced in the monitor-session's
`environment_monitor.json` (NOT the run manifest, which records the
cohort-generation model), the cell config, and each score record's
`model_revision` + identity hash. Known benign-coverage notes (security
review 2026-10-28): deterministic-monitor views collide across episodes
with identical task instances (48 unique rendered prompts across 128
episodes in condition A — correct behavior for a deterministic function,
but effective-distinct-view counts are reported), and one attack episode
of 64 quotes its hidden instruction verbatim inside its own reasoning
(agent-generated evidence in the UNTRUSTED region — the predeclared
no-censorship semantics; the per-episode quoting rate is reported with
the results).

*(Results tables, figures, and the final numbers will be inserted from
the landed archives; nothing in Sections 4-5 cites memory.)*
