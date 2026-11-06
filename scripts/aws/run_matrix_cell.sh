#!/usr/bin/env bash
# One full matrix cell on a single AWS instance (docs/AWS_SETUP.md §5).
#
# Lifecycle: install -> probe&pin -> serve -> generate -> monitor -> analyze
# -> verify -> export -> sync (optional) -> teardown of the server.
# Every stage is resumable; the workflow journal records the chain.
#
# Usage: run_matrix_cell.sh <cell-config.yaml> <serving-spec.yaml> <run-dir>
set -euo pipefail

CELL_CONFIG="${1:?usage: run_matrix_cell.sh <cell-config.yaml> <serving-spec.yaml> <run-dir>}"
SERVING_SPEC="${2:?serving spec required}"
RUN_DIR="${3:?run dir required}"
PORT="${TG_PORT:-8000}"

echo "=== [1/8] environment install (idempotent) ==="
python -m pip install -q -e '.[research,analysis,serving]' || \
  python -m pip install -q -e '.[analysis,serving]'

echo "=== [2/8] admission probe on this tier (records artifact + pins revisions) ==="
python scripts/aws/probe_and_pin.py --config "$CELL_CONFIG" --serving-spec "$SERVING_SPEC" \
  --role agent --port "$PORT"

echo "=== [3/8] launch pinned vLLM server (argv recorded in the launch record) ==="
python -m traceguard.cli serve-model launch --spec "$SERVING_SPEC" \
  --log-dir "$RUN_DIR/serving" --port "$PORT" --out "$RUN_DIR/serving/launch_record.json" &
SERVER_PID=$!
trap 'kill -INT $SERVER_PID 2>/dev/null || true' EXIT

# wait for readiness
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  sleep 5
done

echo "=== [4/8] episode generation ==="
python -m traceguard.cli generate --config "$CELL_CONFIG" --run-dir "$RUN_DIR" \
  --stage pilot --provider auto

echo "=== [5/8] monitoring under all configured conditions ==="
python -m traceguard.cli monitor --config "$CELL_CONFIG" --run-dir "$RUN_DIR" \
  --monitors rules_v1 llm_v1

echo "=== [6/8] analysis (paired contrasts, calibration, equivalence) ==="
python -m traceguard.cli analyze --config "$CELL_CONFIG" --run-dir "$RUN_DIR"

echo "=== [7/8] verification + canonicalize + release export ==="
python -m traceguard.cli verify --run-dir "$RUN_DIR"
python -m traceguard.cli canonicalize --run-dir "$RUN_DIR"
python -m traceguard.cli export --run-dir "$RUN_DIR" \
  --archive "$RUN_DIR.tar.gz"

echo "=== [8/8] cost report + optional S3 sync ==="
python -m traceguard.cli cost-report --ledger-dir "$RUN_DIR/costs"
if [[ -n "${TG_S3_TARGET:-}" ]]; then
  python -m traceguard.cli sync-evidence --run-dir "$RUN_DIR" --target "$TG_S3_TARGET"
fi

echo "cell complete: $RUN_DIR (release tarball + verified evidence index)"
