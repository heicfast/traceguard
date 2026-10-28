#!/usr/bin/env bash
# TRACEGUARD RQ4 online-study landing sequence (SOP-002 §9.4 variant).
# Usage: bash scripts/land_online.sh <1|2|3> <owner/kernel-slug>
# Downloads the online kernel output under the right account, verifies the
# online run locally, runs the RQ4 paired evaluation against the unmonitored
# baseline cohort, and stages the archive directory for commit.
set -euo pipefail

ACCT="${1:?account 1|2|3}"
KERNEL="${2:?owner/kernel-slug}"

REPO=/home/z/my-project/traceguard
DL=/home/z/my-project/downloads/land-online
DEST=$REPO/runs/v6/online_study
PY=/home/z/.venv/bin/python
BASELINE=$REPO/runs/v6/scale_mon_qwen14/run

echo "== landing online study $KERNEL (acct$ACCT)"
rm -rf "$DL"; mkdir -p "$DL"
cp /home/z/my-project/.secrets/kaggle/acct${ACCT}.token ~/.kaggle/access_token
cd "$DL"
export PATH="$PATH:/home/z/.local/bin"
kaggle kernels output "$KERNEL" -p .

RUN_DIR=$(find "$DL/traceguard-output" -maxdepth 4 -type d -name run | head -1)
[ -n "$RUN_DIR" ] || { echo "FAIL: no run dir in output" >&2; exit 1; }
ENV=$(find "$DL/traceguard-output" -name environment.json -path "*online*" | head -1)
echo "== kernel manifest:"
cat "$ENV" | $PY -c "
import json,sys
m=json.load(sys.stdin)
crit={k:m.get(k) for k in ('kernel','pinned_commit','online_rc','gates_passed_count')}
print(json.dumps({**crit,'all_gates':all(g.get('passed') for g in m.get('gates',[]))},indent=1))
assert m.get('online_rc')==0, 'online_rc nonzero'
assert all(g.get('passed') for g in m.get('gates',[])), 'a kernel gate failed'
"

echo "== stage + local verification"
rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$RUN_DIR" "$DEST/run"
cd "$REPO"
$PY -m traceguard.cli verify --run-dir "$DEST/run" | grep -E '"ok"|regrade|problem' | head -4
echo "== canonicalize + re-verify (full chain)"
$PY -m traceguard.cli canonicalize --run-dir "$DEST/run" >/dev/null
$PY -m traceguard.cli verify --run-dir "$DEST/run" | grep -E '"ok"|problem_count'
echo "== export verification"
$PY -m traceguard.cli export --run-dir "$DEST/run" | grep -E 'release_complete|validation_errors'

echo "== RQ4 paired evaluation vs the unmonitored baseline"
$PY scripts/evaluate_online_study.py --online "$DEST/run" --baseline "$BASELINE"

echo "== staged at runs/v6/online_study (archive + commit follow)"
