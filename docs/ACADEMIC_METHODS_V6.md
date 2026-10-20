# TRACEGUARD academic methods and analysis plan

Status: development protocol. This document does **not** freeze a confirmatory
study and does not upgrade pilot findings into confirmatory evidence.

## 1. Scientific target

TRACEGUARD estimates whether an agent's emitted reasoning provides usable,
incremental information about environment-defined unauthorized behavior beyond
the same episode's observable actions. The estimand is conditional on the
tested agent checkpoint, monitor checkpoint, task distribution, trace protocol,
context budget, and monitoring architecture. “CoT is good/bad” is not an
admissible population-level conclusion.

The causal contrast is same-episode and paired:

\[
\Delta_R = R_{C}(\tau_C) - R_{A}(\tau_A),
\]

where \(A\) is actions-only, \(C\) is actions plus emitted reasoning, and each
threshold \(\tau\) is learned once from benign calibration clusters at the
predeclared target FPR. Thresholds are never fitted on holdout or re-estimated
inside holdout bootstrap replicates.

## 2. Why the pilot may differ from published positive results

The result is not assumed to match any paper. OpenAI's broad monitorability
suite reports that CoT access usually outperforms action/output monitoring and
that longer reasoning is generally more monitorable. That suite spans multiple
evaluation archetypes and frontier systems; TRACEGUARD's pilot used one 4B
agent/monitor pairing, a small number of independent clusters, and tasks where
actions often exposed the violation directly.

The relevant competing evidence is also real:

- Monitoring frontier reasoning models found strong CoT-monitoring utility but
  also showed that direct optimization can hide intent while misbehavior stays.
- SHADE-Arena demonstrates long-horizon side objectives and reports that subtle
  sabotage is difficult to monitor.
- MonitoringBench separates attack conception, execution, and transcript
  refinement; its refined attacks substantially reduce catch rate and identify
  persuasion and score calibration as failure modes.
- MonitorBench stresses whether traces expose decision-critical factors rather
  than treating all emitted text as faithful cognition.
- SLEIGHT-Bench explicitly searches for monitor blind spots and tests patches.

TRACEGUARD therefore treats the pilot's self-authorization/persuasion pattern
as a hypothesis to challenge, not a headline to preserve.

## 3. Factorial evaluation structure

The study matrix separates variables that must not be conflated:

| Axis | Levels / role |
|---|---|
| Agent checkpoint | continuity 4B; admitted 14B reasoning model; cross-family robustness |
| Monitor checkpoint | same-scale anchor; scaled same-family; cross-family |
| Evidence | actions; raw CoT; length-matched noise; shuffled CoT; CoT-only |
| Integration | raw concatenation; isolated late fusion; independent fact check |
| Attack presentation | explicit; side objective; implicit pressure; monitor aware |
| Task distribution | development; calibration; held-out family; distribution shift |
| Adaptation | static; taxonomy-diverse; monitor-adaptive development attack |

Generation cohorts are deduplicated so the same immutable episode is scored by
all eligible monitors. Model size, quantization, context cap, and tuning are
recorded factors, never silently pooled.

## 4. Primary and secondary inference

Primary uncertainty uses a paired cluster bootstrap at the template or family
level. The point estimate and every replicate use frozen calibration
thresholds. A confidence interval is suppressed as `not_estimable` when cluster
support, positive-cluster count, bootstrap support, or degenerate-replicate
rules fail.

The analysis also reports an exact McNemar test on paired attack detections.
This test is useful for auditing discordant outcomes but is explicitly
secondary because it does not correct for template clustering. P-values for all
predeclared contrasts within holdout (and separately shift) receive Holm
family-wise-error adjustment. Neither significance nor adjusted significance
overrides the effect estimate, interval, retention, headroom, or calibration
gates.

AUROC and average precision are secondary rank diagnostics. They cannot replace
the fixed-FPR primary estimand after a failed precision gate. Precision-recall
curves are included because attacks may be rare. Score ties are processed as a
group; record order never invents scorer resolution.

## 5. Sizing model

`traceguard study-plan` exposes all design assumptions. For paired binary
detections, let \(q=p_{10}+p_{01}\) be the discordant-pair rate and
\(\delta=p_{10}-p_{01}\) the minimum effect. The independent-pair approximation
is

\[
n \approx \frac{(z_{1-\alpha/2}+z_{power})^2(q-\delta^2)}{\delta^2}.
\]

It is inflated by \(1+(m-1)\rho\), where \(m\) is mean episodes per cluster and
\(\rho\) is the pilot-estimated intracluster correlation. It is then divided by
attack-policy allocation × attack-attempt yield × valid-pair retention to
derive total episodes. Before confirmatory freeze, replace every template
assumption with pilot measurements and validate operating characteristics with
a simulation matching the actual cluster-size distribution. Episode count
alone does not establish power.

## 6. Quality gates and interpretation

A confirmatory claim requires all of the following:

1. pinned code, tokenizer, model, adapter, task registry, and config hashes;
2. independent calibration-cluster and allowed-false-positive gates;
3. action-only headroom (neither floor nor saturation);
4. attack-attempt and honest-task capability by attack stratum;
5. pairwise retention, score resolution, trace coverage, and parse gates;
6. estimable cluster interval and complete censor accounting;
7. no development/holdout family leakage or tuning-example contamination;
8. evidence-index and checksum verification;
9. a signed/timestamped preregistration committed before the holdout run.

Raw CoT losing while isolated fusion wins supports an integration failure.
All CoT arms losing with headroom supports only a bounded negative finding.
Gains limited to explicit attacks do not establish adversarial robustness.
Failures under shuffled/noise controls can reveal length, style, or persuasion
effects. Offline whole-trajectory detection and online pre-action prevention
remain separate protocols and claims.

## 7. Primary references

- Guan et al., *Monitoring Monitorability* (2025), arXiv:2512.18311.
- OpenAI, *Evaluating chain-of-thought monitorability* (2025),
  <https://openai.com/index/evaluating-chain-of-thought-monitorability/>.
- Baker et al., *Monitoring Reasoning Models for Misbehavior and the Risks of
  Promoting Obfuscation* (2025),
  <https://cdn.openai.com/pdf/34f2ada6-870f-4c26-9790-fd8def56387f/CoT_Monitoring.pdf>.
- Greenblatt et al., *AI Control: Improving Safety Despite Intentional
  Subversion* (ICML 2024), arXiv:2312.06942.
- Kutasov et al., *SHADE-Arena* (2025), arXiv:2506.15740.
- Jotautaitė et al., *MonitoringBench* (2026), arXiv:2605.09684.
- Wang et al., *MonitorBench* (2026), arXiv:2603.28590.
- Najt et al., *SLEIGHT-Bench* (2026),
  <https://alignment.anthropic.com/2026/sleight-bench/>.
- NeurIPS, *Paper Checklist Guidelines*,
  <https://neurips.cc/public/guides/PaperChecklist>.
- NIST, *AI Risk Management Framework 1.0* (2023), NIST AI 100-1.

