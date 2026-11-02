#!/usr/bin/env bash
# Closed-loop demo on one machine: Spark controller (local mode) + streaming line simulator.
#   ./scripts/run_demo.sh [steps] [seconds-per-step]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

STEPS="${1:-360}"
INTERVAL="${2:-0.5}"
RUNTIME=$(python3 -c "print(int($STEPS * $INTERVAL + 60))")

rm -rf data/stream data/output data/checkpoints
mkdir -p data/stream/telemetry logs

echo "[demo] starting Spark controller (logs/controller.log)"
python3 -m rac.spark_controller --master "local[4]" --timeout "$RUNTIME" > logs/controller.log 2>&1 &
CTRL_PID=$!
trap 'kill $CTRL_PID 2>/dev/null || true' EXIT

echo "[demo] waiting for Spark to start..."
sleep 25

echo "[demo] streaming $STEPS steps from the assembly line"
python3 -m rac.simulator online --steps "$STEPS" --real-interval "$INTERVAL"

echo "[demo] letting Spark drain the last files..."
sleep 20
kill $CTRL_PID 2>/dev/null || true
wait $CTRL_PID 2>/dev/null || true

python3 -m rac.report
