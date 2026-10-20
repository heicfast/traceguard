# SOP-002 — Kaggle GPU Operations

Standard operating procedure for all GPU execution on Kaggle (kaggle-cli authenticated via `KAGGLE_API_TOKEN`; token stored locally at `~/.kaggle/access_token`, NEVER committed).

## 1. Facts to respect

- Kernel types: `script` (we use scripts, not notebooks — fully deterministic, single `.py` file).
- Accelerator choice via `machine_shape` in `kernel-metadata.json`: `NvidiaTeslaT4` (T4 ×2, default GPU) is preferred; P100 has documented community issues with some 2026 default PyTorch CUDA builds (missing `sm_60` kernels). L4 may be available.
- Internet must be enabled (`enable_internet: true`) for git clone + pip + HF downloads.
- 12-hour execution cap; weekly GPU quota is account-limited and must be checked before large batches.
- Output files are downloaded via `kaggle kernels output <owner>/<slug> -p <dir>`; `/kaggle/working` persists per saved version.

## 2. Kernel preparation checklist (per kernel)

1. Kernel script lives in `kaggle/<name>/kernel.py`; metadata in `kaggle/<name>/kernel-metadata.json`.
2. The script must be a THIN wrapper: clone the repo at a PINNED commit, `pip install -e .`, invoke `python -m traceguard.cli <cmd>`; no experiment logic in the kernel (blueprint rule: code-first, notebook-second).
3. Pinned commit: set `TRACEGUARD_COMMIT` in the script (or metadata) — never "main".
4. The kernel must end by writing `environment.json` (via gpu_preflight or the runtime), checksums, and a completion marker.
5. Secrets: NONE. The public repo is cloned anonymously; no tokens in kernel code or outputs.

## 3. Push / monitor / download loop

```bash
export PATH="$PATH:$HOME/.local/bin"

# push (creates or updates the kernel and starts a run)
kaggle kernels push -p kaggle/<name>

# poll status until complete (QUEUED -> RUNNING -> COMPLETE | ERROR)
kaggle kernels status umar0x/traceguard-<name>

# download all output files
kaggle kernels output umar0x/traceguard-<name> -p downloads/<name> -o

# then locally:
python -m traceguard.cli verify --run-dir downloads/<name>/traceguard-output/<run>
```

Poll cadence: every 3–5 minutes, max 4 hours per poll cycle; record final status + failure message in the worklog. Interruptions and errors are recorded, never retried blindly more than twice (see QUALITY_MANUAL failure ladder).

### 3.1 Queue behavior and manual starts (owner-confirmed facts, 2026-10-20)

- Kaggle kernel queue admission is **erratic**: the same push flow sometimes
  starts instantly and sometimes sits for hours. This is normal Kaggle behavior,
  NOT a quota or account problem.
- **Session pools (verified 2026-10-20):** pushed kernels are *batch* sessions
  (max 2 concurrent per account — QUEUED kernels already hold slots; further
  pushes are rejected with "Maximum batch GPU session count of 2 reached").
  The owner's *interactive* starts run in a separate pool and are admitted
  immediately — which is why cancelling a queued batch version and starting it
  interactively works instantly.
- The project owner may **manually cancel a queued kernel and start it
  interactively** (interactive starts run immediately). Therefore:
  - `CANCEL_ACKNOWLEDGED` on a pushed kernel means "owner intervention" by
    default. Do NOT diagnose quota exhaustion from cancellations.
  - Before re-pushing a kernel, ALWAYS check current statuses of ALL kernels
    first (the owner may already be running a kernel manually).
  - A re-push while 2 batch slots are held is REJECTED — free a slot first or
    wait for the queued kernels to run.
- Standing rule when a push sits queued too long (~20-30+ min): cancel and
  re-push, or let the owner start it interactively. Re-push works.
- GPU allowance is ample (owner-confirmed 25+ hours available as of
  2026-10-20). Never assert quota exhaustion without positive evidence
  (NH-7 applies to infrastructure claims too).

## 4. GPU preflight kernel (`traceguard-preflight`)

Purpose: hardware + torch sanity with REAL tensor operations (not just `torch.cuda.is_available()`), BEFORE downloading model weights.

Checks: PyTorch/CUDA versions (Transformers >=4.51 is checked separately for
Qwen3), device capability, VRAM, fp16 512×512 matmul + reduction finiteness,
and an nvidia-smi dump.

**Gate:** `check: passed` required before any model kernel. If P100 shows non-finite results, switch to T4 shape; do not downgrade CUDA blindly.

## 5. Generation kernel (`traceguard-gen`)

- Clone repo at pinned commit, install deps, resolve Qwen3-4B revision (exact HF commit SHA recorded into episode provenance).
- Run `python -m traceguard.cli generate --config configs/pilot.yaml --run-dir /kaggle/working/traceguard-output/<stage> --stage pilot --shard N --shard-count M`.
- Sharding: M chosen so each shard fits well within 12h (estimate from S2 timing + 40% margin). Resumable: re-pushing the same shard command continues from the ledger.
- End-of-run: write checksums + environment + export bundle.

## 6. Monitoring kernel (`traceguard-mon`)

- Attaches the generation kernel's output as a `kernel_sources` data source (episodes land in `/kaggle/input/`), OR re-downloads via attached dataset — chosen per which artifacts exist.
- Runs `python -m traceguard.cli monitor --config configs/pilot.yaml --run-dir /kaggle/input/<...> --monitors rules_v1 keywords_v1 llm_v1 --conditions actions actions_plus_cot actions_plus_noise actions_plus_shuffled_cot`.
- Writes score shards + checksums. Resumable per (episode, condition, monitor).

## 7. Downloaded-artifact discipline

- Every download is verified (`traceguard verify`) before analysis.
- Downloaded runs are moved under `runs/` (gitignored) locally and their checksums recorded in the worklog.
- Nothing from `/kaggle/working` is considered evidence until it passes verification on local CPU.

## 8. Quota management

- Before each batch: record current quota state (from the account page, noted manually in worklog) and the measured tokens/sec × episodes estimate.
- Prefer fewer, larger shards to reduce queue overhead; keep one spare session for retries.
- If quota is insufficient for the confirmatory design: reduce secondary conditions FIRST (documented in deviations BEFORE the holdout is opened), never the primary comparison.

## 9. Multi-model / three-account execution

- Account quotas add wall-clock capacity only; they do not create pooled VRAM.
- Freeze the sourced registry entries, exact revisions, admission-probe hashes,
  and `configs/model_matrix*.yaml` before scheduling.
- Run `traceguard matrix-plan` and `traceguard matrix-materialize`; the emitted
  `schedule.json` is the authoritative assignment to account slots 0/1/2.
- A generation cohort is produced once per replicate. Every monitor cell that
  names it receives a verified copy of those exact episodes.
- Run one model copy per T4. Use `scripts/run_gpu_workers.py` for generation and
  `scripts/run_monitor_workers.py` for scoring. Never let two processes append
  to one score shard.
- Mount the applicable admission JSON and set the role-specific
  `TRACEGUARD_*_PROBE_ARTIFACT` variable. Runtime hash/identity/gate checks are
  binding.
- Install bitsandbytes only for a declared NF4/INT8 cell. Record its exact
  version; never describe bitsandbytes NF4 as Q4_K_M.
- Preserve each account's original kernel output, logs, environment snapshot,
  monitor-session manifests, and release archive. Merge only through validated
  score/episode identities.
- Account identity is an operational scheduling attribute, not a statistical
  replicate. Replicates are the frozen replicate IDs and seeds in the matrix.

### 9.1 Fleet operational facts (2026-10-20)

- Accounts (scheduling quota only, never statistical replicates):
  `umar0x` (acct1, original), `heicfast` (acct2), `heicjpg` (acct3).
  Auth token files live outside the repository at
  `.secrets/kaggle/acct{1,2,3}.token` (never committed); the CLI reads the
  active token from `~/.kaggle/access_token` (swap via
  `source .secrets/kaggle/use.sh acctN`).
- The repository keeps ONE canonical `kernel-metadata.json` per kernel
  (owner `umar0x`). To push any kernel under a different account, use
  `scripts/kaggle_push.sh <kernel-dir> <acctN>` (workspace-side helper,
  not part of the research package): it copies the kernel dir to a temp
  path, rewrites only the metadata id owner, and pushes. Repo metadata
  never carries account-specific drift.
- Per-account concurrent batch GPU sessions remain capped at 2; queued
  batch kernels occupy their slots. The owner may cancel any queued kernel
  and restart it interactively (interactive runs start immediately and are
  the owner's explicit action — a CANCELLED batch status is never evidence
  of a quota or system problem).
- Current allocation (Phase 3 pilot preparation):
  acct3 `heicjpg/traceguard-episode-generation` (128-episode pipeline pilot,
  anchor cell), acct1 `umar0x/traceguard-probe-qwen14` (monitor-scale
  admission probe), acct2 `heicfast/traceguard-probe-deepseek14`
  (monitor-transfer admission probe).
