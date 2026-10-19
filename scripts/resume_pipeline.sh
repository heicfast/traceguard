#!/usr/bin/env env bash
# TRACEGUARD pipeline resume: re-push queued/cancelled GPU kernels and drive
# the full pilot sequence. Run this when Kaggle GPU quota is available again.
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
MON="umar0x/traceguard-mon"
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
  mkdir -p "$DL/gpu_smoke_v4"
  $KG kernels output "$GEN" -p "$DL/gpu_smoke_v4" && \
    python3 "$HERE/scripts/audit_episodes.py" \
      --run-dir "$DL/gpu_smoke_v4/traceguard-output/gpu_smoke/run" && \
    python3 "$HERE/scripts/pilot_gate.py" \
      --run-dir "$DL/gpu_smoke_v4/traceguard-output/gpu_smoke/run"
  # pilot shard 0
  mkdir -p "$DL/pilot_shard0"
  $KG kernels output "$SHARD0" -p "$DL/pilot_shard0"
}

pilot() {
  push_all
  echo "Polling shard 0 until COMPLETE (GPU queue may take hours)..."
  while true; do
    s=$($KG kernels status "$SHARD0" 2>&1)
    echo "  $(date -u +%H:%M:%S) $s"
    case "$s" in
      *COMPLETE*) break ;;
      *CANCEL*) echo "CANCELLED — quota likely exhausted; retry later with 'pilot'"; exit 2 ;;
    esac
    sleep 300
  done
  collect
  # shard 1 (a slot is free once shard 0 completes)
  ( cd "$HERE/kaggle/gen_pilot_shard1" && $KG kernels push -p . )
  while true; do
    s=$($KG kernels status "$SHARD1" 2>&1)
    echo "  $(date -u +%H:%M:%S) $s"
    case "$s" in
      *COMPLETE*) break ;;
      *CANCEL*) echo "CANCELLED — quota likely exhausted; retry later with 'pilot'"; exit 2 ;;
    esac
    sleep 300
  done
  mkdir -p "$DL/pilot_shard1"
  $KG kernels output "$SHARD1" -p "$DL/pilot_shard1"
  # merge + verify
  python3 "$HERE/scripts/merge_shards.py" \
    --shard-dir "$DL/pilot_shard0/traceguard-output/pilot/run" \
    --shard-dir "$DL/pilot_shard1/traceguard-output/pilot/run" \
    --out "$DL/pilot_merged/run"
  # audit + gates on the merged pilot
  python3 "$HERE/scripts/audit_episodes.py" --run-dir "$DL/pilot_merged/run"
  python3 "$HERE/scripts/pilot_gate.py" --run-dir "$DL/pilot_merged/run"
  # dataset for the monitor kernel
  mkdir -p "$DL/pilot_dataset"
  cp -r "$DL/pilot_merged/run" "$DL/pilot_dataset/traceguard-output/pilot/run"
  cp "$HERE/kaggle/dataset_metadata.json" "$DL/pilot_dataset/dataset-metadata.json"
  ( cd "$DL/pilot_dataset" && $KG datasets create -p . )
  # monitor kernel (scores all conditions; ~3h GPU)
  ( cd "$HERE/kaggle/mon" && $KG kernels push -p . )
  while true; do
    s=$($KG kernels status "$MON" 2>&1)
    echo "  $(date -u +%H:%M:%S) $s"
    case "$s" in
      *COMPLETE*) break ;;
      *CANCEL*) echo "CANCELLED — quota likely exhausted; retry later with 'pilot'"; exit 2 ;;
    esac
    sleep 300
  done
  mkdir -p "$DL/pilot_mon"
  $KG kernels output "$MON" -p "$DL/pilot_mon"
  # analysis (CPU)
  python3 "$HERE/scripts/reproduce_run.py" \
    --run-dir "$DL/pilot_merged/run" --monitor llm_qwen_hf || \
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
