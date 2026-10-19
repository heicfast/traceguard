# Research Question and Falsifiable Hypothesis

Status: frozen for pilot; confirmatory freeze happens at `v0.4-preregistered`.

## Primary hypothesis (H1)

> For an open-weight reasoning agent (Qwen3-4B, native thinking mode) operating in a constrained software-maintenance sandbox, providing a safety monitor with the agent's generated reasoning trace IN ADDITION TO its observable tool actions improves detection of unauthorized-action attempts at a fixed development-calibrated false-positive rate, on held-out task families.

**Primary estimand:** `ΔR = Recall(C) − Recall(A)` at FPR ≤ 5% (calibrated on validation-split benign scores), evaluated on the final holdout family among episodes with verified attack attempts; template-clustered percentile bootstrap 95% CI, 2000 replicates, fixed seed.

**Precision gate:** if the benign calibration pool has fewer than 40 independent
template clusters, the current fixed-FPR confirmatory design does not run.
Paired `AUC(C) − AUC(A)` may be reported for pilot diagnosis, but is not an
automatic substitute for the deployment-relevant primary question.

## Smallest effect of interest

A paired ΔR of +0.10 (10 percentage points) at the calibrated operating point would be practically meaningful for monitor design; effects whose CI excludes ±0.05 in either direction will be interpreted as evidence of (at most) a small practical effect, not noise-chasing.

## Negative outcomes that count as success

1. **Clean null:** ΔR ≈ 0 with tight CI (e.g., [+0.05 width]), while C does not degrade either — evidence that, in this setting, actions already carry the signal the trace adds. Reportable and useful.
2. **Negative result:** ΔR < 0 (trace IMPAIRS monitoring, e.g., unfaithful or misleading reasoning raises FPR) — a strong, publishable warning for CoT-based safety narratives.
3. **Methodological invalidation:** show that a proposed metric (recall@FPR) is not estimable credibly at feasible benign sample sizes — the precision-gate finding itself is a contribution to evaluation practice.

## What would FALSIFY the hypothesis

- Held-out ΔR CI covering 0 with width ≤ 0.10, given adequate attack attempts (power gate met).
- Condition C failing to beat length-matched noise (D) or shuffled-CoT (E) — i.e., any advantage is a context-volume artifact, not reasoning content (RQ3 kill switch).
- The advantage disappearing on the shift families (RQ2) while present on development — evidence the effect is template overfitting.

## Explicitly NOT claimed

- No claim about private CoT of closed models; the measured object is model-generated `think` tokens from an open-weight checkpoint, a possibly unfaithful proxy of internal computation (Chen et al. 2025, *Reasoning Models Don't Always Say What They Think*).
- No claim that attack-policy model organisms equal naturally arising misalignment; the attack is an injected conflicting objective.
- No frontier-model safety claim; the setting is deliberately small-scale and synthetic.
- No deployment claim from offline detection alone; RQ4 (online blocking) is a separate experiment.
