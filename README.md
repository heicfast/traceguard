# TRACEGUARD

**When Does Chain-of-Thought Visibility Improve AI Control? Evaluating Reasoning-Aware Monitors Under Distribution Shift.**

TRACEGUARD is an empirical AI safety research framework studying one bounded question: does access to an agent's generated reasoning trace improve a monitor's ability to identify unauthorized agent behavior, *beyond the information contained in observable tool actions* — and does that improvement survive distribution shift, length/token confounds, and the step from detection to prevention?

This repository contains a **working research engine** (deterministic sandbox, agent runtime, evidence pipeline, monitor conditions, statistical analysis) plus the **frozen protocol** for the study. It is not a product, not a deployable safeguard, and makes no results claims beyond recorded, hashed evidence in `runs/` and released reports.

## Status

| Milestone | State |
|---|---|
| M0 design + governance | done |
| M1 deterministic sandbox + exact grader + tests | done (CPU-only suite) |
| M2 evidence pipeline + view compiler + leakage tests | done |
| M3 FakeModel end-to-end CPU integration | done |
| M4 Kaggle GPU preflight + Qwen3 integration | done (T4, fp16, SDPA; all gates passed) |
| M5 monitor conditions A/C/D/E on same episodes | done |
| M6 64-episode pilot + independent review | done; exploratory, negative-leaning; **full evidentiary archive committed** (`runs/pre-tests-data/pilot/` — episodes, privileged records, transactions, 768 monitor scores, analysis report, figures, per-file SHA-256 manifest; `traceguard export` validates it fail-closed) |
| M6.5 Phase 3 hardening | **merged (PR #3, `2cc54d0`)** — relational non-action-visible families, 4 attack-presentation strata, 0–100 rubric, structured late fusion, pairwise-complete analysis |
| M6.6 multi-model/T4 architecture | **merged (PR #3, `2cc54d0`)** — audited registry, fail-closed probe-admission, dual-T4 workers, release contract |
| M6.7 Phase 3 pilot (pre-test) | **generated, verified, gate-passed, archived** — 128 episodes (`runs/pre-tests-data/phase3/pilot_v0_run`), attempts 83%, attack success 100%, honest completion 98%, trace coverage 100%; public dataset `heicjpg/traceguard-phase3-pilot-run-v1`; **superseded by the v0.6.0 from-scratch restart (owner directive `5d1a00e`)** |
| M6.8 monitor-capability cells (pre-test) | matrix frozen + scoring started under v0.5.x; **stopped by owner directive and folded into the v0.6.0 restart** |
| M6.9 v0.6.0 core upgrade + multi-POV review | **merged; 3 independent reviews; ALL findings fixed** — 2 BLOCKERs (gate retries; export re-certifying tampered archives) + 14 MAJORs; 218 tests, ruff + mypy clean (`c3fe999`) |
| M6.10 v0.6.0 from-scratch program | **IN FLIGHT** — GPU smoke running; all four fresh 14B role-admissions PASSED (`runs/v6/probes/`); fresh anchor cohort next; pre-test data renamed to `runs/pre-tests-data/` |
| M7 preregistration freeze | blocked on Phase 3 gates and sizing |
| M8 confirmatory + online intervention | pending |

**Current stage: v0.6.0 from-scratch program (owner directive `5d1a00e`).**
The owner upgraded the research core (workflow engine with hash-chained journal,
evidence index, failure taxonomy, McNemar+Holm analysis, sizing model, figure
contract) and directed a full restart: all in-flight runs stopped, all prior data
renamed to `runs/pre-tests-data/`, every phase re-executed on the hardened core
with fresh admissions. The v0.6 core passed three independent expert reviews with
every BLOCKER/MAJOR fixed before any run (`c3fe999`, 218 tests). Fresh program
identity: `traceguard_v6_pilot_v1`. All four 14B role-admissions passed on v0.6
(`runs/v6/probes/`); the GPU smoke and the fresh 128-episode anchor cohort
are complete (attempts 83%, success 53/53, honest 63/64, re-grade 128/128).

**Phase-3 progress (2026-10-21).** The full 5-cell model-capability matrix
is executing on the three-account fleet with the v2 monitor instrument:
anchor (4B monitor over the 4B cohort), scale (Qwen3-14B monitor), cross
(DeepSeek-R1-14B monitor), and the two agent-cohort cells. Both agent
cohorts are landed, verified (64/64 re-grade each), archived
(`runs/v6/agent_cohorts/`) and published as datasets. Headline agent
result: stronger agents are MORE reliable attackers AND better honest
workers — Qwen3-14B attempts 32/32 (100%) with 100% honest completion;
DeepSeek-R1-14B 29/32 (91%); vs 83% attempts for the 4B anchor.
The v2 instrument (attempt-1 post-mortem, `docs/deviations.md`
2026-10-21): the model now reports four banded rubric components only and
the parser derives the 0-100 score deterministically — the v1 contract's
arithmetic self-report invalidated 69-89% of 4B monitor outputs without
touching the underlying rubric judgments.

**Pilot result, most defensible statement** (exploratory, 64 episodes,
Qwen3-4B agent × same-checkpoint monitor): adding capped CoT did not improve
— and appeared to degrade — the actions-only baseline on the holdout split
(recall 1.0 → 0.0 under strict preregistered censoring; ≈0.33 under
channel-repaired parsing; 2 benign calibration clusters vs 40 required, so
all fixed-FPR statements are exploratory). Raw traces show pervasive
self-authorization framing; recorded monitor verdicts show several cases
where the scorer adopted that framing while citing the violating action. The
effect is confounded by small effective sample size, action-visible attacks,
coarse scores, differential JSON failures, censoring, and generic
long-context degradation — exactly the competing explanations Phase 3 is
designed to separate.

**Phase 3 review response:** the kernel is suitable for a new pilot, but confirmatory
execution is deliberately blocked. The current validation split has only two
independent template clusters versus the predeclared minimum of forty for a
fixed-5%-FPR estimate, and the task bank must pass an action-only saturation
gate. See [`docs/RESEARCH_REVIEW_V2.md`](docs/RESEARCH_REVIEW_V2.md).

## The four non-negotiable rules

1. **No predetermined conclusion.** The hypothesis is that CoT *may* improve monitoring. The framework must be able to show that it doesn't, and a null/negative result is a valid outcome.
2. **No fake ground truth.** Unauthorized-action labels come from exact state-transition checks against protected-invariant predicates (the environment decides), never from a language-model judge.
3. **No benchmark leakage.** Monitor prompts, thresholds, and analysis choices are frozen before the final held-out evaluation; splits are family-scoped and hash-pinned.
4. **No GPU dependency for core correctness.** Sandbox, grading, view compilation, statistics, and all tests run on a local CPU. GPU (Kaggle) is used only for model inference.

## Quick start (CPU, no GPU required)

```bash
python -m pip install -e '.[analysis,dev]'
python -m pytest -q                          # full test suite
python -m traceguard.cli doctor             # registry, split-manifest, config self-check
python -m traceguard.cli smoke --out runs/smoke   # full pipeline smoke with scripted agent
```

## Research questions

- **RQ1 (marginal monitoring value):** At a development-calibrated false-positive rate, does adding the reasoning trace to action evidence improve recall of unauthorized-action attempts on held-out episodes? Primary estimand: `ΔR = Recall(actions+CoT) − Recall(actions)`, paired per episode, template-clustered bootstrap CI.
- **RQ2 (distribution shift):** Does any advantage survive on entirely held-out task families and a held-out environment shift set?
- **RQ3 (information controls):** Is the advantage attributable to reasoning *content* (vs. more tokens / generic text / unrelated traces), via length-matched noise (D) and shuffled-CoT (E) conditions?
- **RQ4 (control effectiveness, phase two):** When the monitor acts *before* action execution (fresh online rollouts, no replays), does it reduce completed violations without unacceptable loss of legitimate task completion?

## Architecture in one paragraph

Frozen experiment configs drive seeded task-instance generation across twelve synthetic task families. A deterministic in-memory sandbox executes strictly typed tool calls (`inspect_state`, `list_objects`, `propose_patch`, `submit_task`) and judges direct and relational protected-invariant violations by exact state comparison. The Qwen runtime records complete prompt and generated token ids, decoded outputs, messages, seeds, revisions, chat-template/runtime metadata, and hashes; CPU tests use a scripted fake. Episode evidence is committed through a crash-recoverable transaction before JSONL projection. An allowlist ViewCompiler builds raw, control, late-fusion, and fact-checked conditions with mechanical leakage assertions. Monitors retain exact prompts, raw responses, structured intermediate reports, and inference audits. The analysis layer uses comparison-specific pairing, frozen thresholds, CI-eligibility gates, exact small-cluster distributions, missing-score bounds, and provenance-stamped figures.

## Repository layout

```text
runs/pre-tests-data/pilot/pilot_v0_run/   COMPLETE pre-test evidentiary archive (64 episodes):
  episodes/ + privileged/ + transactions/ + completed_episode_ids.txt
  monitor_scores/          768 scores: 3 monitors x 4 conditions, every record
                           carries rendered prompts, raw responses, inference
                           audits, and per-record hashes
  reports/                 analysis_report.json + figures (+ plot data)
  artifacts/               per-file SHA-256 manifest + checksums (release gate)
runs/pre-tests-data/pilot/   pre-test kernel logs + monitor-stage environment manifest
runs/gpu_smoke_v*/         earlier validated smoke runs (same structure)
docs/reviews/              multi-POV pilot reviews (scientist, security,
                           statistician) with independently verified findings
src/traceguard/
  tasks/       TaskSpec registry (12 families, 24 templates), seeded instances, split manifest
  sandbox/     pure state, typed actions, engine, invariants, exact grader
  models/      provider contract, audited model registry, quantized HF adapters
  agents/      honest/attack policy composition, episode runner
  traces/      schema validation, atomic storage, allowlist view compiler
  monitors/    rules/keyword baselines, LLM monitor, condition harness
  protocols/   pre-action online intervention (phase two)
  analyses/    metrics (AUROC/AP/recall@FPR), paired cluster bootstrap, plots
  cli.py       doctor | probe-model | matrix-plan/materialize | prepare-tuning | pipeline
configs/       smoke/pilot configs plus the multi-model matrix freeze template
tests/         unit + property + integration tests
docs/          protocol, architecture, SOPs, quality manual, review system
kaggle/        GPU kernel scripts + metadata (preflight, generation, monitoring)
```

## Governance (read before running anything)

- [`docs/QUALITY_MANUAL.md`](docs/QUALITY_MANUAL.md) — no-fallback and no-hallucination policy, verification cadence, evidence rules
- [`docs/SOP_001_EXPERIMENT_EXECUTION.md`](docs/SOP_001_EXPERIMENT_EXECUTION.md) — stage-by-stage execution procedure with gates
- [`docs/SOP_002_KAGGLE_OPERATIONS.md`](docs/SOP_002_KAGGLE_OPERATIONS.md) — Kaggle push/monitor/download operations
- [`docs/REVIEW_SYSTEM.md`](docs/REVIEW_SYSTEM.md) — multi-POV independent review protocol (developer / scientist / security / statistician)
- [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) — trusted-computing boundary, data flow, performance and hard gates
- [`docs/RAW_EVIDENCE_SPEC.md`](docs/RAW_EVIDENCE_SPEC.md) — exact token/prompt/runtime retention and release separation
- [`docs/RESEARCH_REVIEW_V2.md`](docs/RESEARCH_REVIEW_V2.md) — consolidated findings and confirmatory blockers
- [`docs/PHASE3_PROTOCOL.md`](docs/PHASE3_PROTOCOL.md) — harder task strata, structured monitoring, statistics, and release contract
- [`docs/MULTIMODEL_STUDY.md`](docs/MULTIMODEL_STUDY.md) — T4 feasibility, audited model matrix, admission probes, tuning isolation, and raw-evidence additions
- [`docs/ACADEMIC_METHODS_V6.md`](docs/ACADEMIC_METHODS_V6.md) — literature-grounded estimands, factorial design, sizing mathematics, multiplicity, and claim rules
- [`docs/AUTOMATION_AND_EVIDENCE_V6.md`](docs/AUTOMATION_AND_EVIDENCE_V6.md) — resumable workflow journal, failure taxonomy, evidence index, and figure contract
- [`docs/prior_art_matrix.md`](docs/prior_art_matrix.md) — how TRACEGUARD differs from SLEIGHT-Bench, MonitorBench, OpenAI monitorability evals, SHADE-Arena
- [`docs/research_question.md`](docs/research_question.md) — the one falsifiable hypothesis and its negative outcomes
- [`docs/deviations.md`](docs/deviations.md) — dated deviation log (changes after any freeze)

## Grounding

- Korbak et al., *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety* — https://arxiv.org/abs/2507.11473
- Greenblatt et al., *AI Control: Improving Safety Despite Intentional Subversion* https://proceedings.mlr.press/v235/greenblatt24a.html
- Najt, Toft, Tracy, Roger, Benton, *SLEIGHT-Bench: A Benchmark of Evasion Attacks Against Agent Monitors* (arXiv:2605.16626) — https://alignment.anthropic.com/2026/sleight-bench/
- OpenAI, *Evaluating chain-of-thought monitorability* — https://openai.com/index/evaluating-chain-of-thought-monitorability
- Kutasov et al., *Evaluating Control Protocols for Untrusted AI Agents* (AAAI 2026 workshop)
- Chen et al., *Reasoning Models Don't Always Say What They Think* — https://arxiv.org/abs/2505.05410
- Inspect AI — https://inspect.aisi.org.uk/ · ControlArena — https://control-arena.aisi.org.uk/
- Qwen3-4B — https://huggingface.co/Qwen/Qwen3-4B (Apache 2.0)
- Wang et al., *MonitorBench* — https://arxiv.org/abs/2603.28590
- OpenAI, *Reasoning Models Struggle to Control their Chains of Thought* — https://openai.com/index/reasoning-models-chain-of-thought-controllability/
- 

## License

Apache-2.0. All data in this repository is synthetic. No real credentials, hosts, repositories, or user accounts are referenced by the simulation.
