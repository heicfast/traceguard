# 05 — Research Quality, Ethics, and Publication Standards

## 1. Scientific honesty

This project is proposal-stage. No numeric result, published paper, working Kaggle notebook, model performance guarantee, or validated novelty is claimed in this package. Label pilots as pilots and simulations as simulations. The paper's wording must distinguish *the behavior of agents given an explicit attack objective* from *spontaneous deceptive alignment*. These are not equivalent claims.

## 2. Threats to validity, with countermeasures

| Threat | Why results might mislead | Countermeasure |
|---|---|---|
| Toy-world artifacts | Agents/monitors exploit simple path names | Randomize identifiers; vary semantics; hold out families |
| Artificially obvious attacker | Attack prompts openly reveal side task | Hide attack setup from monitor, audit naturalness, quantify attack quality |
| Incapable attacker | No unsafe action is attempted | Separate attack intention / attempt / success; capability gate |
| Perfect rule baseline | CoT adds no value on trivially checkable rules | Report null honestly; design harder new tasks separately |
| Model-generated trace unfaithfulness | Trace may omit causal factors or be strategic | Avoid intent claims, add text-matched controls, targeted interventions later |
| More tokens in CoT condition | Better score from context volume alone | Length-matched noise, scrambled CoT, log truncation |
| Leakage of private instructions | Monitor detects attack metadata rather than behavior | Strict allowlist view compiler; adversarial leak tests |
| Template duplication | Held-out episodes too similar to training | Split entire families and report near-duplicate audit |
| Base-rate distortion | Balanced eval looks better than rare-event deployment | Present prevalence-adjusted precision illustrations as assumptions, not deployment estimates |
| Poor low-FPR precision | 1% FPR impossible to measure well at tiny N | Use feasibility target (e.g. 5%) and exact uncertainty; scale benign test if claiming 1% |
| Monitor compute confounding | Trace monitor gets more output budget | Fix monitor settings; log latency/tokens; stratify cost |
| Confirmation bias | Prompts and metrics changed after seeing failures | Preregister, freeze configs, maintain deviations log |
| Online/offline conflation | Catching after a bad action does not prevent it | Strict pre-action protocol and fresh rollouts for intervention |
| Dependence / pseudo-replication | Episodes share template / agent models | Template-clustered bootstrap and grouped splits |
| Kaggle nondeterminism | GPU kernels/sampling vary across runs | Exact seeds, logs, model SHA, independent repeats where feasible |
| Hidden dataset imbalance | Attack and benign episodes differ in unrelated ways | Paired task specifications and negative controls |

## 3. Safety and responsible release

All environments are synthetic and offline. Tool invocation is a typed data structure interpreted only by a pure transition engine. Never add real shell, network, credential, cloud, or data-exfiltration tools to the untrusted model's experiment sandbox. Use fake IDs and fake hosts rather than reachable destinations. Treat agent text as untrusted; do not execute generated snippets. Scan and review traces for accidental personal or secret data before publication.

Publishing attacker traces can be dual-use; evaluate whether a particular artifact reveals transferable evasion techniques, and if so consider publishing higher-level measurements while restricting detailed examples. State the rationale and request an independent release review before public publication.

## 4. Reproducibility package

At minimum each release includes:

- release Git commit, license, pinned dependencies, exact model and tokenizer version;
- Kaggle runtime summary and hardware type, failure and OOM counts;
- task definitions and frozen splits, if safe to share;
- monitor prompt sources, compiled-view rules, and hash of evidence observed;
- dataset creation scripts with all RNG seeds;
- generated episode and monitor score artifacts or a meaningful, safely shared reproduction subset;
- exact analysis notebooks/scripts and data dictionary;
- CIs and denominators for all estimates;
- detailed limitations and deviation log;
- one reproducibility command producing the primary plot from archived evidence.

If the raw generated episodes cannot be shared due to safety or copyright issues, disclose this and state which claims remain independently verifiable.

## 5. Documentation and engineering review

Use type checking, linting, unit tests, deterministic integration tests and explicit error states. No silent exception swallowing or auto-relabeling; preserve all failure/error counts. A successful notebook cell does not constitute an experiment result if the required manifest is missing. Peer reviews must check both model assumptions and code correctness.

## 6. Paper outline (target 8–12 pages + appendix)

1. **Abstract:** one question, one controlled design, one quantified outcome, one limitation.
2. **Introduction:** why trace visibility might help but can be fragile.
3. **Related work:** monitorability, AI control, SLEIGHT, open-source eval suites, faithful reasoning.
4. **Threat model:** deployment boundary and what untrusted/trusted roles know.
5. **Methods:** tasks, organisms, monitor conditions, controls, scoring, splits.
6. **Results:** primary paired effect, denominator and confidence interval; honest/attack capability funnel.
7. **Ablations:** token-matched/noise/shuffle controls and held-out family.
8. **Control experiment:** online prevention utility (if run).
9. **Discussion:** why failures occur, what does not generalize, limitations of generated traces.
10. **Conclusion:** evidence-supported narrow claim, not global safety assurance.
11. **Appendices:** prompts, seeds, configurations, formulae, audit examples, failure counts.

Do not fabricate a “significant” result or remove underperforming baselines. A careful null/negative result often provides a stronger portfolio signal than a flashy, unreliable claim.

## 7. Release/versioning policy

- `v0.1-design`: blueprint, schemas, no results.
- `v0.2-engine`: CPU simulation and tests complete.
- `v0.3-pilot`: pilot and quality audit.
- `v0.4-preregistered`: frozen confirmatory protocol.
- `v0.5-results`: immutable held-out artifacts and analysis.
- `v1.0-paper`: complete report and reproduction release.

Any change that affects labels, task design, model prompts, metric definitions or thresholds after freeze requires a dated deviation. A substantially revised experiment should use a new preregistration and new holdout.

## 8. Mentorship / external review

Request feedback on the *specific claim and methodology*, not generic praise. Useful questions: Is the negative control strong enough? What independent observations are confounded? Would a simple deterministic monitor solve this task? Is task-family holdout genuinely hard? Does the paper overstate relevance to frontier-model scheming? Report substantive criticism and changes in a public issue or changelog where appropriate.
