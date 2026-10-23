#!/usr/bin/env bash
# Focused bisect: which component is necessary for the deterministic fault?
# Uses the frozen reproducer (--no_optimizer, fault window ~iter 26-49),
# 100 iters per config (covers the fault window with margin).
set -u
cd "$(dirname "$0")"
RESULTS=bisect_results.txt
echo "config,result,last_iter" > "$RESULTS"

run_config() {
  local label="$1"; shift
  local log="logs/bisect_${label}.log"
  python3 -u reproducer.py --no_optimizer --iters 100 "$@" > "$log" 2>&1
  if grep -q "Memory access fault" "$log"; then r=FAULT
  elif grep -q COMPLETED "$log"; then r=COMPLETED
  else r=OTHER; fi
  local last=$(grep -oE "iter +[0-9]+" "$log" | tail -1)
  echo "$label,$r,$last" >> "$RESULTS"
  echo "$label: $r ($last)"
}

run_config baseline
run_config no_moe --first_k_dense 6
run_config no_mtp --n_mtp 0
run_config one_layer --n_layer 1
run_config two_layers --n_layer 2
run_config fp32 --dtype fp32
run_config experts4 --n_experts 4
run_config uniform_routing --n_experts_per_tok 8
run_config batch16 --batch_size 16
run_config block64 --block_size 64 --sliding_window 32
run_config mha --n_kv_head 8
run_config no_swa --sliding_window 256 --full_attn_every 1
run_config dense_control --impl dense
run_config cpu_control --device cpu --n_layer 2 --n_embd 128 --batch_size 16 --block_size 64 --iters 20

echo "=== bisect done ===" >> "$RESULTS"
cat "$RESULTS"
