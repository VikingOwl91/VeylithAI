#!/usr/bin/env bash
# Phase D scaling grid: sparse vs dense across batch sizes, sequence
# lengths, and expert configurations. OOM is recorded as an outcome, not
# silently skipped.
set -u
cd "$(dirname "$0")/.."   # veylith/debug
PY="${HOME}"/veylith-debug-venvs/phase4/bin/python
RESULTS=version-isolation/scaling-results.txt
echo "config,impl,fwd_ms,bwd_ms,step_ms,tok_s,alloc_GB,reserved_GB" > "$RESULTS"

run() {
  local label="$1"; shift
  for impl in sparse dense; do
    local log="logs/scaling_${label}_${impl}.log"
    $PY -u benchmark.py --impl $impl "$@" > "$log" 2>&1
    local code=$?
    if grep -q "OutOfMemoryError" "$log"; then
      echo "$label,$impl,OOM,,,," >> "$RESULTS"
      echo "$label $impl: OOM"
      continue
    fi
    if [ $code -ne 0 ]; then
      echo "$label,$impl,FAILED(exit=$code),,,," >> "$RESULTS"
      echo "$label $impl: FAILED (exit=$code): $(tail -1 "$log" | head -c 120)"
      continue
    fi
    local fwd bwd step toks alloc res
    fwd=$(grep -oE "forward  ms: median=[0-9.]+" "$log" | grep -oE "[0-9.]+$")
    bwd=$(grep -oE "backward ms: median=[0-9.]+" "$log" | grep -oE "[0-9.]+$")
    step=$(grep -oE "full step ms: median=[0-9.]+" "$log" | grep -oE "[0-9.]+$")
    toks=$(grep -oE "tokens/sec \(median step\): [0-9,]+" "$log" | grep -oE "[0-9,]+$" | tr -d ',')
    alloc=$(grep -oE "peak GPU allocated: [0-9.]+" "$log" | grep -oE "[0-9.]+$")
    res=$(grep -oE "reserved: [0-9.]+" "$log" | grep -oE "[0-9.]+$")
    echo "$label,$impl,$fwd,$bwd,$step,$toks,$alloc,$res" >> "$RESULTS"
    echo "$label $impl: fwd=${fwd}ms bwd=${bwd}ms step=${step}ms tok/s=$toks alloc=${alloc}GB"
  done
}

run b16_blk256   --batch_size 16 --block_size 256
run b64_blk256   --batch_size 64 --block_size 256
run b128_blk256  --batch_size 128 --block_size 256
run b64_blk512   --batch_size 64 --block_size 512
run b128_blk512  --batch_size 128 --block_size 512
run b64_e16k4    --batch_size 64 --block_size 256 --n_experts 16 --n_experts_per_tok 4

echo "=== scaling done ===" >> "$RESULTS"
cat "$RESULTS"
