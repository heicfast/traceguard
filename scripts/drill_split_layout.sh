#!/bin/bash
# Local drill of the SINGLE-SPLIT worker layout (2026-10-27 attempt-3 fix).
# Exercises the exact new code path end-to-end on CPU: topology planning ->
# per-worker env (both devices visible + device_map=auto) -> single-worker
# spawn -> score merge -> checksum rewrite. Deterministic monitors only
# (rules/keywords need no provider, no CUDA); --visible-gpus 2 keeps the
# parent off torch while the topology still plans a two-GPU split.
# Also drills per_gpu for score-count equivalence, and the fail-closed
# single_split-with-1-GPU refusal.
set -euo pipefail
cd /home/z/my-project/traceguard
PY=/home/z/.venv/bin/python3
DRILL=/tmp/split_layout_drill
rm -rf "$DRILL"; mkdir -p "$DRILL"

# Config reconstructed from the run manifest (placeholder-free), matching
# the mon kernel pattern and the runtime consistency gate.
$PY - << 'PYEOF'
import json, yaml
man = json.load(open("runs/v6/pilot_v0_run/run_manifest.json"))
with open("/tmp/split_layout_drill/config.yaml", "w") as f:
    f.write(yaml.safe_dump(man["config"], sort_keys=True))
print("config reconstructed from manifest")
PYEOF

echo "== [0] pure topology check =="
$PY - << 'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "rmw", "scripts/run_monitor_workers.py")
mod = importlib.util.module_from_spec(spec); sys.modules["rmw"] = mod
spec.loader.exec_module(mod)
split = mod.plan_worker_topology(2, "single_split")
assert len(split) == 1 and split[0]["visible_devices"] == "0,1"
assert split[0]["device_map"] == "auto" and split[0]["shard_count"] == 1
per_gpu = mod.plan_worker_topology(2, "per_gpu")
assert [s["visible_devices"] for s in per_gpu] == ["0", "1"]
assert [s["shard"] for s in per_gpu] == [0, 1]
print("topology OK: single_split spans both devices with one worker")
PYEOF

for LAYOUT in single_split per_gpu; do
  echo "== [1] $LAYOUT layout over a fresh copy of the 128-episode cohort =="
  rm -rf "$DRILL/run_$LAYOUT"
  mkdir -p "$DRILL/run_$LAYOUT"
  cp -r runs/v6/pilot_v0_run "$DRILL/run_$LAYOUT/run"
  rm -rf "$DRILL/run_$LAYOUT/run/artifacts/checksums.sha256"
  $PY scripts/run_monitor_workers.py \
      --config "$DRILL/config.yaml" \
      --run-dir "$DRILL/run_$LAYOUT/run" \
      --layout "$LAYOUT" --visible-gpus 2 \
      --monitors rules_v1 keywords_v1 2>&1 | tail -9
done

echo "== [2] equivalence + structure assertions =="
$PY - << 'PYEOF'
import glob, json
def load(layout):
    scores = {}
    for f in glob.glob(f"/tmp/split_layout_drill/run_{layout}/run/monitor_scores/*.jsonl"):
        for line in open(f):
            rec = json.loads(line)
            scores[rec["score_id"]] = rec
    return scores
split_scores = load("single_split")
gpu_scores = load("per_gpu")
assert len(split_scores) == len(gpu_scores) > 0, (len(split_scores), len(gpu_scores))
# Same score IDs and same score values across layouts (deterministic monitors)
for sid, rec in split_scores.items():
    assert gpu_scores[sid] == rec, f"layout drift at {sid}"
workers_split = glob.glob("/tmp/split_layout_drill/run_single_split/run-monitor-workers/gpu-*")
assert len(workers_split) == 1, f"single_split must spawn exactly ONE worker, saw {workers_split}"
workers_gpu = glob.glob("/tmp/split_layout_drill/run_per_gpu/run-monitor-workers/gpu-*")
assert len(workers_gpu) == 2, f"per_gpu must spawn two workers, saw {workers_gpu}"
print(f"equivalence OK: {len(split_scores)} identical scores; 1 split worker vs 2 per_gpu workers")
PYEOF

echo "== [3] fail-closed: single_split with one visible GPU must refuse =="
set +e
$PY scripts/run_monitor_workers.py --config "$DRILL/config.yaml" \
    --run-dir "$DRILL/run_single_split/run" --layout single_split --visible-gpus 1 \
    --monitors rules_v1 > "$DRILL/refusal.json" 2>&1
rc=$?
set -e
[ "$rc" -eq 1 ] || { echo "FAIL: expected rc=1, got $rc"; exit 1; }
grep -q "FORBIDDEN" "$DRILL/refusal.json" && echo "refusal OK (no fallback)"

echo "== [4] integrity chain on the single_split-scored copy =="
$PY -m traceguard.cli canonicalize --run-dir "$DRILL/run_single_split/run" 2>&1 | tail -2
$PY -m traceguard.cli verify --run-dir "$DRILL/run_single_split/run" > "$DRILL/verify.json" 2>&1 \
  || { echo "verify FAILED:"; tail -20 "$DRILL/verify.json"; exit 1; }
$PY -c "import json; d=json.load(open('$DRILL/verify.json')); print('verify ok:', d.get('ok', d))"

echo "SINGLE-SPLIT DRILL COMPLETE"
