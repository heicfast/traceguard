#!/usr/bin/env bash
# TRACEGUARD cell-landing helper (SOP-002 §9.4), updated 2026-10-21:
# - per-account auth via scripts/kgl.sh (access-token method)
# - robust resumable download (scripts/download_kernel_output.sh)
# - kernel-manifest critical checks include canonicalize_rc (v2 chain)
# Downloads a completed cell kernel's output under the right account, verifies
# the scored run locally (engine verify), and stages the run for archiving.
# Never mutates the downloaded original; works on a copy.
#
# Usage: bash scripts/land_cell.sh <1|2|3> <owner>/<kernel-slug> <dest-name>
set -euo pipefail

ACCT="${1:?account 1|2|3}"
KERNEL="${2:?owner/kernel-slug}"
DEST="${3:?archive dir name under runs/v6/}"

REPO=/home/z/my-project/traceguard
DL=/home/z/my-project/downloads/land-${DEST}
PY=/home/z/.venv/bin/python3

echo "== landing $KERNEL (acct$ACCT) -> runs/v6/$DEST"
rm -rf "$DL"; mkdir -p "$DL"

bash /home/z/my-project/scripts/download_kernel_output.sh "$ACCT" "$KERNEL" "$DL"
if [ ! -d "$DL/traceguard-output" ]; then
  echo "FAIL: no traceguard-output in kernel output" >&2; exit 1
fi

RUN_DIR=$(find "$DL/traceguard-output" -maxdepth 3 -type d -name run | head -1)
if [ -z "$RUN_DIR" ]; then
  echo "FAIL: no run dir under traceguard-output" >&2; exit 1
fi
echo "== scored run at: $RUN_DIR"

$PY - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
env = run / "environment_monitor.json"
if not env.exists():
    print("FAIL: environment_monitor.json missing", file=sys.stderr); sys.exit(1)
m = json.loads(env.read_text())
keys = ("kernel", "cell_id", "pinned_commit", "monitor_rc", "analysis_rc",
        "canonicalize_rc", "monitor_gate_rc", "verify_rc", "export_rc", "kernel_seconds")
print("kernel manifest:", json.dumps({k: m.get(k) for k in keys}, indent=1))
crit = [m.get("monitor_rc"), m.get("analysis_rc"), m.get("canonicalize_rc"),
        m.get("verify_rc"), m.get("export_rc")]
if any(c != 0 for c in crit):
    print("FAIL: kernel critical commands nonzero:", crit, file=sys.stderr); sys.exit(1)
PY

echo "== local verification on a copy"
DEST_DIR=$REPO/runs/v6/$DEST
rm -rf "$DEST_DIR"; mkdir -p "$DEST_DIR"
cp -r "$RUN_DIR" "$DEST_DIR/run"
cd "$REPO"
$PY -m traceguard.cli verify --run-dir "$DEST_DIR/run" | grep -E '"ok"|regrade|problem' | head -3
SCORE_COUNT=$(find "$DEST_DIR/run/monitor_scores" -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
echo "== score records: $SCORE_COUNT"
echo "== staged at runs/v6/$DEST (gate + analysis check + archive + commit follow)"
