#!/bin/bash
# Local end-to-end drill of the mon-kernel chain on the archived fresh v6
# cohort (deterministic monitors only; the LLM arm needs GPU).
# Proves: score -> analyze -> CANONICALIZE -> verify -> export all pass
# with the 2026-10-21 chain fix, on the EXACT failure mode that killed
# anchor-mon attempt 1 (analysis-written reports breaking strict verify).
set -euo pipefail
cd /home/z/my-project/traceguard
PY=/home/z/.venv/bin/python3
DRILL=/tmp/mon_chain_drill
rm -rf "$DRILL"; mkdir -p "$DRILL"
cp -r runs/v6/pilot_v0_run "$DRILL/run"

# Same pattern as the mon kernel: the config is reconstructed from the run
# manifest (placeholder-free), which is also what the runtime consistency
# gate requires.
$PY - << 'PYEOF'
import json, yaml
man = json.load(open("runs/v6/pilot_v0_run/run_manifest.json"))
with open("/tmp/mon_chain_drill/config_from_manifest.yaml", "w") as f:
    f.write(yaml.safe_dump(man["config"], sort_keys=True))
print("config reconstructed from manifest")
PYEOF
CFG="$DRILL/config_from_manifest.yaml"

echo "== [1] deterministic monitors over 128 episodes =="
# (worker orchestration is the kernel-side GPU path; locally we run the
#  monitor CLI directly — rules/keywords need no provider)
$PY -m traceguard.cli monitor --config "$CFG" \
    --run-dir "$DRILL/run" --monitors rules_v1 keywords_v1 \
    --shard 0 --shard-count 1 2>&1 | tail -8

echo "== [2] analyze (writes reports/ + tables/ INTO the run dir) =="
$PY -m traceguard.cli analyze --config "$CFG" \
    --run-dir "$DRILL/run" --monitor rules_v1 2>&1 | tail -4

echo "== [3] verify BEFORE canonicalize (must FAIL with uncovered reports — reproducing the kernel bug) =="
set +e
$PY -m traceguard.cli verify --run-dir "$DRILL/run" > "$DRILL/verify_before.json" 2>&1
rc=$?
set -e
echo "verify-before rc=$rc (expected nonzero)"
grep -c "not covered by checksum ledger" "$DRILL/verify_before.json" || true

echo "== [4] canonicalize (NEW step: index first, then ledger covering the index) =="
$PY -m traceguard.cli canonicalize --run-dir "$DRILL/run"

echo "== [5] verify AFTER canonicalize (must PASS) =="
$PY -m traceguard.cli verify --run-dir "$DRILL/run" | tail -6

echo "== [6] export release bundle =="
$PY -m traceguard.cli export --run-dir "$DRILL/run" --archive "$DRILL/release.tar.gz" | tail -4

echo "DRILL COMPLETE: all chain steps behave as designed"
