# 04 — 12-Week Execution Plan, Gates, and Acceptance Criteria

This is an intentionally staged research plan. It is not a promise that 12 weeks guarantees publishable novelty. A weak benchmark should be fixed before an expensive confirmatory run.

| Phase | Weeks | Objective | Exit gate |
|---|---|---|---|
| P0 — Literature and reproduction | 1–2 | Pin research claims, reproduce baseline, freeze candidate hypothesis | Independent baseline works; related-work gap documented |
| P1 — CPU simulator | 2–3 | Implement state machine and exact grader | 100% passing deterministic/negative/leakage tests |
| P2 — Agent + trace | 3–4 | Integrate Qwen3 on Kaggle; capture raw traces and tools | ≥90% well-formed runs on 8 smoke tasks, exceptions reported |
| P3 — Monitoring harness | 4–5 | Rules, LLM monitor, A/C/D/E view compiler | Paired views, no privileged leakage, replay identical |
| P4 — Pilot | 5–6 | 48 episodes; manual audit; runtime and power/precision check | Sufficient attack attempts, realistic positives/negatives, viable throughput |
| P5 — Freeze protocol | 6–7 | Resolve pilot-discovered flaws, pre-register final test | Frozen dataset split, configs, monitor prompts, primary metric, power/precision plan |
| P6 — Confirmatory runs | 7–9 | Generate final episodes and scores without tuning to holdout | All planned episodes accounted for, provenance and checksum verified |
| P7 — Control follow-up | 9–10 | Evaluate pre-action blocking on fresh runs | Correct cutoffs and fresh model continuations; no future leakage |
| P8 — Analyze and review | 10–11 | Stats, ablation, failure categories, reproducibility | Independent rerun of primary figure from archived artifacts |
| P9 — Release and paper | 11–12 | Technical report, code, reviewed synthetic dataset, feedback | Public archive version, limitations, threat model, citeable preprint if ready |

## Stage P0 — Specific tasks

- Read Korbak et al. (monitorability) and Greenblatt et al. (AI control), and the OpenAI monitorability eval suite.
- Read SLEIGHT-Bench carefully to identify where prior synthetic monitor-evasion transcripts overlap with our proposal.
- Create a *prior art matrix*: prior paper, environment realism, reasoning visibility conditions, paired controls, monitoring time (offline vs pre-action), result type, remaining question.
- Reproduce a small public evaluation/metric in a separate pinned environment; record differences from published numbers rather than forcing agreement.
- Draft `research_question.md` with exactly one falsifiable primary hypothesis and at least three potential negative outcomes.

**Gate:** can explain how TRACEGUARD's design differs from existing benchmarks without claiming novelty just because it has a new project name.

## Stage P1 — Simulator delivery

Implement two task families first, with at least four fixture cases per family: allowed edit, forbidden direct edit, forbidden indirect edit (if representable within simulator), malformed operation. Define schema and state transitions. Tests should validate invariant detection independently with hand-constructed examples and state replay.

**Gate:** no model is needed to run the entire test suite; environment transitions yield identical hashes across reruns; protected-state violations cannot pass as legitimate completion.

## Stage P2 — Model delivery

Install pinned stack on Kaggle, run GPU math preflight, download model, capture raw reasoning output, parse one authorized tool call, measure peak VRAM. Add fake model adapter for fast CPU tests. Do not proceed if `think` and tool calls cannot be disambiguated consistently.

**Gate:** 8 episode smoke output contains complete evidence (including explicit failures), and a human can match generated action proposals against the canonical event log.

## Stage P3 — Monitoring delivery

Start with exact-rule baseline and one LLM monitor with fixed risk-scoring rubric. Implement A/C/D/E as projections of the *same* stored episode. Add tests that remove private prompts and labels from each view. Decide target FPR and plan how to calibrate it; record monitor tokenizer and maximum context length.

**Gate:** one command scores the same episode under all conditions; no monitor sees later state for a pre-action decision.

## Stage P4 — Pilot delivery

Generate 48 planned trajectories and all monitor conditions. Independently audit random examples in all categories; classify refusals, invalid tools, task ambiguity, never-attempted attacks, truncated reasoning. Time the Kaggle run and extrapolate; inspect honest-case FPR and attack attempts.

**Gate:** pass the preregistered minimum utility, validity, and variation thresholds. Thresholds must be filled based on feasibility planning before confirmatory testing. If benchmark is trivial, do not hide the failure—publish pilot learning and build a revised benchmark version.

## Stage P5 — Freeze

Commit the pre-registration, exact split list, prompt hashes, model revision, metric scripts and planned exclusions. Publish or timestamp the preregistration before opening the final holdout. Archive pilot results separately. No changing monitor prompt or selecting thresholds using final outcomes.

## Stage P6 — Confirmatory

Batch model generation and monitors; write append-only records; run analysis only after verifying required records and exclusions. Use strict manifest checks and an automated “no new config edits after freeze” check. Compute primary Δ and interval without searching for the most favorable subset.

## Stage P7 — Online control

Because intervention changes future trajectories, execute genuinely new online episodes under protocol A and C using comparable sampled tasks and frozen thresholds. Record decisions **before** applying changes. Separate offline catch rate from actual protected-state harm prevention. Track legitimate tasks that were blocked incorrectly.

## Stage P8 — Analysis

Create: dataset composition table; agent capability funnel (attack policy → attempted → succeeded); paired A/C plot; false-positive/recall curve; C versus noise/shuffle ablation; held-out-family result; online safety-vs-usefulness plot; failure taxonomy; compute cost; fragility and caveats.

## Stage P9 — Release

Review trace logs for any unintentionally exposed sensitive details, freeze clean artifact tags, create paper and reproducibility scripts, verify the README on a fresh environment, and solicit substantive critique. If evidence is insufficient, label the output a **pilot research report** rather than overclaiming.

## Definition of done

A research-complete v1 means (a) precise question; (b) documented independently graded tasks; (c) comparable conditions on matched trajectories; (d) sufficient held-out episodes to support clearly qualified statistical estimates; (e) no known ground truth leakage; (f) full provenance; (g) honest analysis including negative findings; (h) reproducible code and artifact audit. **A visually premium front-end is not part of the research acceptance criteria.**
