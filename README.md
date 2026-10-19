# TRACEGUARD

**When Does Chain-of-Thought Visibility Improve AI Control? Evaluating Reasoning-Aware Monitors Under Distribution Shift.**

TRACEGUARD is an empirical AI safety research framework studying one bounded question: does access to an agent's generated reasoning trace improve a monitor's ability to identify unauthorized agent behavior, *beyond the information contained in observable tool actions* — and does that improvement survive distribution shift, length/token confounds, and the step from detection to prevention?

This repository contains a **working research engine** (deterministic sandbox, agent runtime, evidence pipeline, monitor conditions, statistical analysis) plus the **frozen protocol** for the study. It is not a product, not a deployable safeguard, and makes no results claims beyond recorded, hashed evidence in `runs/` and released reports.

## Status

| Milestone | State |
|---|---|
| M0 design + governance | done |
| M1 deterministic sandbox + exact grader + tests | done (95 tests, CPU-only) |
| M2 evidence pipeline + view compiler + leakage tests | done |
| M3 FakeModel end-to-end CPU integration | done |
| M4 Kaggle GPU preflight + Qwen3 integration | see `kaggle/` + `worklog/` |
| M5 monitor conditions A/C/D/E on same episodes | code done; GPU runs pending |
| M6 pilot + capability/precision gate | pending GPU |
| M7 preregistration freeze | template ready |
| M8 confirmatory + online intervention | pending |

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

Frozen experiment configs drive seeded task-instance generation across six synthetic task families (config updates, schema migration, dependency metadata, incident workflow, service rotation, backup policy). A deterministic in-memory sandbox executes strictly typed tool calls (`inspect_state`, `list_objects`, `propose_patch`, `submit_task`) and judges protected-invariant violations by exact state comparison. The agent runtime (Qwen3-4B in native thinking mode on Kaggle, or a scripted fake for CPU tests) records every turn: genuine `think` tokens, visible answers, tool proposals, tool results, state hashes. An allowlist ViewCompiler compiles monitor evidence (conditions A/B/C/D/E/F) with mechanical leakage assertions; hidden instructions, outcome labels, and `hidden_instruction` events can never enter monitor views. Deterministic baselines (rules, keywords) and an LLM monitor with a frozen prompt score every condition on the *same* stored episodes. The analysis layer computes paired recall differences with template-clustered bootstrap confidence intervals, precision gates for low-FPR claims, capability funnels, and provenance-stamped figures.

## Repository layout

```
src/traceguard/
  tasks/       TaskSpec registry (6 families, 12 templates), seeded instances, split manifest
  sandbox/     pure state, typed actions, engine, invariants, exact grader
  models/      provider contract, FakeModel/scripted agent, Qwen3 HF adapter
  agents/      honest/attack policy composition, episode runner
  traces/      schema validation, atomic storage, allowlist view compiler
  monitors/    rules/keyword baselines, LLM monitor, condition harness
  protocols/   pre-action online intervention (phase two)
  analyses/    metrics (AUROC/AP/recall@FPR), paired cluster bootstrap, plots
  cli.py       doctor | smoke | generate | monitor | analyze | verify | export
configs/       smoke.yaml, pilot.yaml (frozen candidates)
tests/         unit + property + integration (95 tests)
docs/          protocol, architecture, SOPs, quality manual, review system
kaggle/        GPU kernel scripts + metadata (preflight, generation, monitoring)
```

## Governance (read before running anything)

- [`docs/QUALITY_MANUAL.md`](docs/QUALITY_MANUAL.md) — no-fallback and no-hallucination policy, verification cadence, evidence rules
- [`docs/SOP_001_EXPERIMENT_EXECUTION.md`](docs/SOP_001_EXPERIMENT_EXECUTION.md) — stage-by-stage execution procedure with gates
- [`docs/SOP_002_KAGGLE_OPERATIONS.md`](docs/SOP_002_KAGGLE_OPERATIONS.md) — Kaggle push/monitor/download operations
- [`docs/REVIEW_SYSTEM.md`](docs/REVIEW_SYSTEM.md) — multi-POV independent review protocol (developer / scientist / security / statistician)
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

## License

Apache-2.0. All data in this repository is synthetic. No real credentials, hosts, repositories, or user accounts are referenced by the simulation.
