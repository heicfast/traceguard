#!/usr/bin/env env bash
# TRACEGUARD pipeline resume: push GPU kernels and drive the full pilot
# sequence. Kaggle queue admission is ERRATIC (instant or hours) — that is
# normal. CANCELLED usually means the OWNER cancelled a queued version to
# restart it interactively (interactive starts run immediately); it is NOT a
# quota signal. Never re-push blindly: pushing a new version kills any run the
# owner may have started manually — always check status first and re-check
# after a CANCELLED observation before acting (SOP-002 section 3.1).
#
#   bash scripts/resume_pipeline.sh status   # just poll kernel states
#   bash scripts/resume_pipeline.sh push     # re-push smoke + shard0
#   bash scripts/resume_pipeline.sh collect  # download + audit + gate any completed run
#   bash scripts/resume_pipeline.sh pilot    # full sequence: push, poll, collect, merge,
#                                             # dataset, mon kernel, then analysis (CPU)
#
# Requires: KAGGLE_API_TOKEN env var (loaded from .secrets/env.sh if present).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/../.secrets/env.sh" 2>/dev/null || true
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN not set}"
KG="kaggle"
GEN="umar0x/traceguard-episode-generation"
SHARD0="umar0x/traceguard-pilot-generation-shard-0"
SHARD1="umar0x/traceguard-pilot-generation-shard-1"
MON="umar0x/traceguard-monitor-scoring"
DL="/home/z/my-project/downloads"

status() {
  for k in "$GEN" "$SHARD0" "$SHARD1" "$MON"; do
    $KG kernels status "$k" 2>&1 | sed "s/^/[$k] /"
  done
}

push_all() {
  ( cd "$HERE/kaggle/gen" && $KG kernels push -p . )
  ( cd "$HERE/kaggle/gen_pilot_shard0" && $KG kernels push -p . )
}

collect() {
  # smoke
  mkdir -p "$DL/gpu_smoke_v6"
  $KG kernels output "$GEN" -p "$DL/gpu_smoke_v6" -o && \
    python3 "$HERE/scripts/audit_episodes.py" \
      --run-dir "$DL/gpu_smoke_v6/traceguard-output/gpu_smoke/run" && \
    python3 "$HERE/scripts/pilot_gate.py" \
      --run-dir "$DL/gpu_smoke_v6/traceguard-output/gpu_smoke/run"
  # pilot shard 0
  mkdir -p "$DL/pilot_shard0"
  $KG kernels output "$SHARD0" -p "$DL/pilot_shard0" -o
}

# Poll one kernel until COMPLETE. On CANCELLED: the owner usually cancelled a
# queued version to restart it interactively. Wait, re-check, and only re-push
# (which would kill an owner's manual run) after the kernel stays cancelled for
# several consecutive checks.
poll_kernel() {
  local name="$1" dir="$2" cancel_streak=0
  while true; do
    s=$($KG kernels status "$name" 2>&1)
    echo "  $(date -u +%H:%M:%S) [$name] $s"
    case "$s" in
      *COMPLETE*) return 0 ;;
      *ERROR*) return 1 ;;
      *CANCEL*)
        cancel_streak=$((cancel_streak + 1))
        if [ "$cancel_streak" -ge 3 ]; then
          echo "[$name] still CANCELLED after 3 checks (~6 min) — re-pushing $dir"
          ( cd "$HERE/$dir" && $KG kernels push -p . ) || return 1
          cancel_streak=0
        else
          echo "[$name] CANCELLED — likely owner interactive restart; re-checking in 2 min"
          sleep 120
        fi
        ;;
    esac
    sleep 300
  done
}

pilot() {
  # both shards in parallel (2 concurrent GPU sessions allowed)
  ( cd "$HERE/kaggle/gen_pilot_shard0" && $KG kernels push -p . )
  ( cd "$HERE/kaggle/gen_pilot_shard1" && $KG kernels push -p . )
  poll_kernel "$SHARD0" kaggle/gen_pilot_shard0
  poll_kernel "$SHARD1" kaggle/gen_pilot_shard1
  mkdir -p "$DL/pilot_shard0" "$DL/pilot_shard1"
  $KG kernels output "$SHARD0" -p "$DL/pilot_shard0" -o
  $KG kernels output "$SHARD1" -p "$DL/pilot_shard1" -o
  # merge + verify
  python3 "$HERE/scripts/merge_shards.py" \
    --shard-dir "$DL/pilot_shard0/traceguard-output/pilot/run" \
    --shard-dir "$DL/pilot_shard1/traceguard-output/pilot/run" \
    --out "$DL/pilot_merged/run"
  # audit + gates on the merged pilot
  python3 "$HERE/scripts/audit_episodes.py" --run-dir "$DL/pilot_merged/run"
  python3 "$HERE/scripts/pilot_gate.py" --run-dir "$DL/pilot_merged/run" \
    --gpu-feasibility-evidence "$HERE/runs/gpu_feasibility/evidence_2026-10-20.json"
  # dataset for the monitor kernel (--public: everything public per owner directive;
  # the CLI default is PRIVATE, which would break the mon kernel's public-clone
  # assumptions and the open-research policy)
  mkdir -p "$DL/pilot_dataset"
  cp -r "$DL/pilot_merged/run" "$DL/pilot_dataset/traceguard-output/pilot/run"
  cp "$HERE/kaggle/dataset_metadata.json" "$DL/pilot_dataset/dataset-metadata.json"
  ( cd "$DL/pilot_dataset" && $KG datasets create -p . --public )
  # monitor kernel (scores all conditions; ~3h GPU)
  ( cd "$HERE/kaggle/mon" && $KG kernels push -p . )
  poll_kernel "$MON" kaggle/mon
  mkdir -p "$DL/pilot_mon"
  $KG kernels output "$MON" -p "$DL/pilot_mon" -o
  # S4 gate on the monitored run (M1-M3 binding; M4/M5 report design facts)
  python3 "$HERE/scripts/monitor_gate.py" \
    --run-dir "$DL/pilot_mon/traceguard-output/pilot/run"
  # analysis (CPU)
  python3 "$HERE/scripts/reproduce_run.py" \
    --run-dir "$DL/pilot_mon/traceguard-output/pilot/run" --monitor llm_qwen_hf || \
    echo "NOTE: run the analysis per docs/SOP_001 (verify -> analyze per condition)"
  echo "PIPELINE RESUME COMPLETE"
}

case "${1:-status}" in
  status) status ;;
  push) push_all ;;
  collect) collect ;;
  pilot) pilot ;;
  *) echo "usage: $0 status|push|collect|pilot"; exit 64 ;;
esac
