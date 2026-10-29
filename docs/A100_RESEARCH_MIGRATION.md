# A100 migration and research upgrade: audit and launch gates

Status: planning and CPU-tested reliability changes only. No A100 inference,
performance benchmark, new-model admission, or confirmatory study is claimed.
The v6 data and its experimental conditions remain historical evidence.

## Immediate execution reliability

Offline monitoring commits each schema-validated score atomically before the
next inference call. A terminated process can resume from committed scores,
provided the same run directory is retained. Each worker still needs a separate
run directory; concurrent writers to the same score shard are unsupported.
The checksum ledger and monitor-session manifest are finalized at the end of a
successful invocation. An interrupted directory is an operational checkpoint,
not a verified or releasable archive: resume it, finalize, and run verification.
Persistent or remote storage must itself survive instance termination; atomic
local writes alone cannot make ephemeral storage durable. Per-score atomic
rewrites can increase disk I/O as shards grow; measure on the rental storage.

The study planner now counts all GPUs in a job when reporting GPU-hours. Its
throughput values describe the entire job, and its sizing remains an
approximation. Historical estimates should be recalculated; no historical
evidence files are modified.

## A100 launch blockers

The existing `probe-model` requires SM 7.5 and FP16 and is therefore a T4
admission probe, not a general GPU probe. The registry also enforces its
T4-recommended quantization even on other hardware. The default per-device
memory budget is 14 GiB. Consequently, do not claim that this branch enables
the 14B FP16 or 32B model path on A100. Existing probe artifacts and frozen
T4 experiments must not be silently reinterpreted as A100 admissions.

Before a new A100 study, implement explicit hardware profiles and probe
artifacts bound to model and tokenizer revisions, precision, backend, context,
generation settings, device type and memory budget. Exercise exact deployment
prompts, long-context memory, reasoning parsing, tool use and rubric output;
test all devices. Require a fresh probe per materially different configuration.
Preserve the T4 path and its existing frozen evidence. Admission is a safety
and measurement gate, not a scientific validity certificate.

## Prioritized research program

1. Finish and verify the existing exploratory cross-monitor and online cells;
   keep their thresholds, splits and model conditions frozen.
2. Develop new, independently varied tasks that require contextual or
   intent-dependent distinctions. Check for leakage and deterministic grading.
   A stronger model cannot fix a task inventory with an actions-only ceiling.
3. Measure action-only headroom and attempted-positive yield on development
   data; create independent benign calibration clusters, not repeated seeds
   relabeled as independent templates. Run power simulations with the intended
   cluster sizes and effect size before freezing the holdout.
4. Preserve the 4B continuity arm; add 14B precision sensitivity and a 32B
   monitor only after distinct admission probes. Evaluate different model
   families as separate arms. Match episodes across monitors, and avoid
   confounding model, quantization and inference backend in causal claims.
5. Use predeclared paired contrasts, calibration-only thresholds, cluster-aware
   uncertainty, invalid/missing-output accounting, task-family holdouts and
   explicit exploratory labels. Keep fresh online intervention separate from
   offline detection and report benign task costs and blocking outcomes.
6. Benchmark realistic short and long agent, raw-monitor, structured-monitor,
   and online examples. Record wall time, GPU count, prompt/output tokens,
   model load time, VRAM and failure rates by condition. Estimate workload by
   stage and sample stratum, then reserve for retries and long-tail episodes.
   Do not derive an A100 speedup from nominal FLOPS or aggregate T4 timings.

## Sources informing the upgrade

- SLEIGHT-Bench emphasizes blind spots and adaptive attack presentations:
  https://alignment.anthropic.com/2026/sleight-bench/
- vLLM documents scheduling and reproducibility caveats for inference:
  https://docs.vllm.ai/en/v0.13.0/usage/reproducibility/
- The project's own protocol and headroom records remain the primary source
  for its estimands, gates and observed exploratory outcomes:
  `docs/PHASE3_PROTOCOL.md`, `docs/ACADEMIC_METHODS_V6.md`, and
  `runs/v6/headroom_by_family_20261028.txt`.

Changing a backend, batching schedule, precision or hardware may change model
outputs. Treat each as an explicit experimental factor and validate evidence
capture and reproducibility before production research runs. No kernel speedup
is claimed without measurements on representative hardware.
