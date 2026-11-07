#!/bin/sh
# Ground-truth queue: one worker, nice 19, pause on battery; every FDTD run takes the machine-wide
# heavy-job lock (tools/heavy_guard.py: normal memory pressure to start, 50% pacing, footprint guard).
# Every stage skips finished work, so the queue can be restarted after a failure.
#   sh docs/benchmarks/method-compare/run_queue.sh [stage ...]
# Stages: build shoebox gt refine assemble (default: all, in that order).
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY="$ROOT/.context/fdtd-teacher/venv/bin/python"
SUITE="$ROOT/tools/method_compare_suite.py"
set -e
cd "$ROOT"
STAGES=${*:-"build shoebox gt refine assemble"}
for stage in $STAGES; do
  echo "$stage $(date +%H:%M:%S)"
  case $stage in
    build) nice -n 19 "$PY" "$SUITE" build > /dev/null ;;
    reflection|shoebox|refine) nice -n 19 "$PY" "$SUITE" validate "$stage" ;;
    gt) nice -n 19 "$PY" "$SUITE" gt ;;
    assemble) nice -n 19 "$PY" "$SUITE" assemble ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "queue finished $(date +%H:%M:%S)"
