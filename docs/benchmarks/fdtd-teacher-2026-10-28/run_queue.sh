#!/bin/sh
# Run study cases one at a time at low priority; stop on the first failure.
# Usage: sh run_queue.sh CASE...   (logs in .context/fdtd-teacher/logs)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$ROOT/.context/fdtd-teacher/logs"
for c in "$@"; do
  if [ -f "$ROOT/.context/fdtd-teacher/runs/$c.npz" ]; then echo "skip $c"; continue; fi
  echo "start $c $(date +%H:%M:%S)"
  nice -n 19 "$ROOT/.context/fdtd-teacher/venv/bin/python" "$ROOT/docs/benchmarks/fdtd-teacher-2026-10-28/study.py" "$c" \
    > "$ROOT/.context/fdtd-teacher/logs/$c.log" 2>&1 || { echo "FAILED $c"; exit 1; }
  echo "done $c $(date +%H:%M:%S)"
  sleep 20
done
echo "queue finished"
