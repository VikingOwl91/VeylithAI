#!/usr/bin/env bash
# Run the version-isolation test suite for one environment.
# Usage: ./run-env-tests.sh <label> <env-python>
set -u
cd "$(dirname "$0")/.."   # veylith/debug
LABEL="$1"; PY="$2"
LOGDIR="version-isolation/logs"
mkdir -p "$LOGDIR"

echo "=== [$LABEL] smoke test ==="
"$PY" version-isolation/smoke.py > "$LOGDIR/${LABEL}_smoke.log" 2>&1
echo "smoke exit: $?"

for i in 1 2 3; do
  "$PY" reproducer.py --no_optimizer --grad_filter experts --iters 100 \
      > "$LOGDIR/${LABEL}_repro_${i}.log" 2>&1
  code=$?
  if grep -q "Memory access fault\|HSA_STATUS_ERROR" "$LOGDIR/${LABEL}_repro_${i}.log"; then r=FAULT
  elif grep -q COMPLETED "$LOGDIR/${LABEL}_repro_${i}.log"; then r=COMPLETED
  else r="OTHER(exit=$code)"; fi
  last=$(grep -oE "iter +[0-9]+" "$LOGDIR/${LABEL}_repro_${i}.log" | tail -1)
  echo "repro run $i: $r (exit=$code, last: $last)"
done

CUDA_LAUNCH_BLOCKING=1 "$PY" reproducer.py --no_optimizer --grad_filter experts --iters 100 \
    > "$LOGDIR/${LABEL}_repro_lb.log" 2>&1
if grep -q "Memory access fault\|HSA_STATUS_ERROR" "$LOGDIR/${LABEL}_repro_lb.log"; then r=FAULT
elif grep -q COMPLETED "$LOGDIR/${LABEL}_repro_lb.log"; then r=COMPLETED
else r=OTHER; fi
echo "repro launch-blocking: $r"
