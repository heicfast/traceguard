#!/usr/bin/env bash
# TRACEGUARD cell-landing helper (SOP-002 §9.4).
# Downloads a completed cell kernel's output under the right account, verifies
# the scored run locally (engine verify + monitor gate), and stages the run for
# archiving. Never mutates the downloaded original; works on a copy.
#
# Usage: bash scripts/land_cell.sh <acctN> <owner>/<kernel-slug> <dest-name>
# Example: bash scripts/land_cell.sh acct2 heicfast/traceguard-mon-cross-deepseek14 cross-monitor-qwen4-deepseek14
set -euo pipefail

ACCT="${1:?account (acctN)}"
KERNEL="${2:?owner/kernel-slug}"
DEST="${3:?archive dir name under runs/v6/}"

SECRETS=/home/z/my-project/.secrets/kaggle/use.sh
REPO=/home/z/my-project/traceguard
DL=/home/z/my-project/downloads/land-${DEST}

# shellcheck source=/dev/null
source "$SECRETS" "$ACCT"
echo "== landing $KERNEL (auth: $ACCT) -> runs/v6/$DEST"

rm -rf "$DL"
mkdir -p "$DL"
cd "$DL"
kaggle kernels output "$KERNEL" 2>&1 | tail -1
if [ ! -d "$DL/traceguard-output" ]; then
  echo "FAIL: no traceguard-output in kernel output" >&2
  exit 1
fi

# locate the scored run copy (kernel writes traceguard-output/<stage>/run)
RUN_DIR=$(find "$DL/traceguard-output" -maxdepth 3 -type d -name run | head -1)
if [ -z "$RUN_DIR" ]; then
  echo "FAIL: no run dir under traceguard-output" >&2
  exit 1
fi
echo "== scored run at: $RUN_DIR"
python3 - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
env = run / "environment_monitor.json"
if env.exists():
    m = json.loads(env.read_text())
    print("kernel manifest:", json.dumps({k: m.get(k) for k in
        ("kernel", "cell_id", "pinned_commit", "monitor_rc", "analysis_rc",
         "monitor_gate_rc", "verify_rc", "export_rc", "kernel_seconds")}, indent=1))
    crit = [m.get("monitor_rc"), m.get("analysis_rc"), m.get("verify_rc"), m.get("export_rc")]
    if any(c != 0 for c in crit):
        print("FAIL: kernel critical commands nonzero:", crit, file=sys.stderr)
        sys.exit(1)
else:
    print("FAIL: environment_monitor.json missing", file=sys.stderr)
    sys.exit(1)
PY

echo "== local verification on a copy"
STAGE=$(basename "$(dirname "$RUN_DIR")")
DEST_DIR=$REPO/runs/v6/$DEST
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"
cp -r "$RUN_DIR" "$DEST_DIR/run"
cd "$REPO"
python3 -m traceguard.cli verify --run-dir "$DEST_DIR/run" | tail -4
SCORE_COUNT=$(find "$DEST_DIR/run/monitor_scores" -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
echo "== score records: $SCORE_COUNT"
echo "== staged at runs/v6/$DEST (archive + commit remain manual)"
