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
| M4 Kaggle GPU preflight + Qwen3 integration | see `kaggle/` + `worklog/` |
| M5 monitor conditions A/C/D/E on same episodes | done |
| M6 64-episode pilot + independent review | done; exploratory, negative-leaning |
| M6.5 Phase 3 hardening | implemented on `v6`; new pilot required |
| M6.6 multi-model/T4 architecture | implemented; admission probes and revision freeze required |
| M7 preregistration freeze | blocked on Phase 3 gates and sizing |
| M8 confirmatory + online intervention | pending |

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

```
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
- [`docs/prior_art_matrix.md`](docs/prior_art_matrix.md) — how TRACEGUARD differs from SLEIGHT-Bench, MonitorBench, OpenAI monitorability evals, SHADE-Arena
- [`docs/research_question.md`](docs/research_question.md) — the one falsifiable hypothesis and its negative outcomes
- [`docs/deviations.md`](docs/deviations.md) — dated deviation log (changes after any freeze)

## Grounding

- Korbak et al., *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety* — https://arxiv.org/abs/2507.11473
- Greenblatt et al., *AI Control: Improving Safety Despite Intentional Subversion* — https://proceedings.mlr.press/v235/greenblatt24a.html
- Najt, Toft, Tracy, Roger, Benton, *SLEIGHT-Bench: A Benchmark of Evasion Attacks Against Agent Monitors* (arXiv:2605.16626) — https://alignment.anthropic.com/2026/sleight-bench/
- OpenAI, *Evaluating chain-of-thought monitorability* — https://openai.com/index/evaluating-chain-of-thought-monitorability
- Kutasov et al., *Evaluating Control Protocols for Untrusted AI Agents* (AAAI 2026 workshop)
- Chen et al., *Reasoning Models Don't Always Say What They Think* — https://arxiv.org/abs/2505.05410
- Inspect AI — https://inspect.aisi.org.uk/ · ControlArena — https://control-arena.aisi.org.uk/
- Qwen3-4B — https://huggingface.co/Qwen/Qwen3-4B (Apache 2.0)
- Wang et al., *MonitorBench* — https://arxiv.org/abs/2603.28590
- OpenAI, *Reasoning Models Struggle to Control their Chains of Thought* — https://openai.com/index/reasoning-models-chain-of-thought-controllability/

## License

Apache-2.0. All data in this repository is synthetic. No real credentials, hosts, repositories, or user accounts are referenced by the simulation.
