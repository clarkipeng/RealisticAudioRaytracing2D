#!/bin/sh
# Pilot queue: one worker at a time, nice 19, 20% pacing, battery pause.
# Stages skip finished scenes, so the queue can be restarted after a failure.
#   sh docs/benchmarks/denoise-pilot-2026-10-29/run_queue.sh
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY="$ROOT/.context/fdtd-teacher/venv/bin/python"
JOBS="$ROOT/tools/dlss_jobs.py"
set -e
cd "$ROOT"
echo "generate $(date +%H:%M:%S)"
nice -n 19 "$PY" "$JOBS" generate --count 12
echo "tracer $(date +%H:%M:%S)"
nice -n 19 "$PY" "$JOBS" render --batch 6
echo "tracer-rigid $(date +%H:%M:%S)"
nice -n 19 "$PY" "$JOBS" render --batch 6 --variant tracer-rigid --fdtd-subset 24
echo "fdtd $(date +%H:%M:%S)"
nice -n 19 "$PY" "$JOBS" fdtd --select 24
echo "queue finished $(date +%H:%M:%S)"
