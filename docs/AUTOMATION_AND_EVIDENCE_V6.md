# Automated execution and evidence architecture

## 1. Design goals

The automation layer is deliberately smaller than a general workflow engine.
It provides the properties this study needs: deterministic argv execution,
declared dependencies and outputs, bounded retry/timeout behavior, resumability,
tamper-evident events, per-attempt logs, and no shell-string evaluation.

## 2. Workflow lifecycle

```mermaid
flowchart TD
    P[Validate workflow] --> D[Dry-run plan]
    D --> R[Run or resume]
    R --> J[Hash-chained event journal]
    J --> O{Outputs exist and hashes match?}
    O -- yes --> S[Skip verified stage]
    O -- no --> X[Execute stage]
    X --> V[Verify, index, export]
```

Use `configs/workflow_phase3.template.yaml` as a reviewed starting point:

```bash
traceguard workflow plan \
  --workflow configs/workflow_phase3.template.yaml --work-dir .
traceguard workflow run \
  --workflow configs/workflow_phase3.template.yaml --work-dir .
traceguard workflow resume \
  --workflow configs/workflow_phase3.template.yaml --work-dir .
traceguard workflow status \
  --workflow configs/workflow_phase3.template.yaml --work-dir .
```

`resume` and `run` intentionally share semantics: a completed stage is skipped
only when its prior success record and every declared output digest still
match. Editing an output makes the stage incomplete. A lock prevents two
orchestrators from writing the same journal concurrently.

## 3. Failure contract

Failures use stable classes: configuration, input integrity, model admission,
resource exhaustion, transient provider, data quality, statistical gate,
software defect, cancellation, and unknown. Every failure includes the stage,
exception type, retryability decision, attempt, and log path. Retries are
bounded in the workflow file. Scientific gate failures are not transient and
must never be retried into a pass by changing the threshold or data.

## 4. Run directory contract

| Directory | Authority | Required content |
|---|---|---|
| `episodes/` | monitor-visible | exact projected events and trace boundaries |
| `privileged/` | ground truth/audit | raw prompts, outputs, token IDs, states, transitions |
| `transactions/` | recovery | write-ahead envelope + privileged pair |
| `commits/` | recovery | transaction digest and projection digests |
| `monitor_scores/` | measurement | prompts, raw responses, parse result, view hash, timing |
| `reports/` | derived | analysis report and plot data |
| `logs/` | operations | per-stage/attempt stdout and diagnostics |
| `artifacts/` | integrity | release manifest, evidence index, checksum ledger |

The evidence index assigns every file a role, media type, size, and SHA-256,
then hashes the ordered inventory. Its derivation contract distinguishes
ground truth, monitor input, measurement, and derived outputs. `verify` checks
the index when present; `export` rebuilds it before final checksums.

## 5. Figures and tables

Analysis emits plot data as JSON next to each figure. Every figure is rendered
as PNG for review, SVG for editable publication graphics, and PDF for archival
use. The standard set contains ROC, precision-recall, score distribution,
paired-effect forest, capability funnel, and measurement completeness views. A
provenance footer links run ID and config digest.

## 6. Validation tiers

- Tier 0: config/schema parsing and `doctor`.
- Tier 1: pure metric fixtures, property tests, journal tamper tests.
- Tier 2: deterministic CPU smoke, re-grade, view leakage, resume behavior.
- Tier 3: model admission on the exact GPU/runtime and exact checkpoint.
- Tier 4: exploratory pilot with every gate and raw artifact retained.
- Tier 5: preregistration freeze, clean confirmatory generation, blind analysis.

The CI workflow runs Tier 0–2 on Python 3.11 and 3.13. GPU admission and real
model evaluation cannot be represented by a green CPU CI badge and stay
separate blocking artifacts.

