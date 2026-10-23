#!/usr/bin/env bash
# Test matrix for the ROCm sparse-MoE memory access fault.
# Drives reproducer.py through a list of configurations and records
# pass/fault per run. Usage: ./run_matrix.sh [runs_per_config]
set -u
cd "$(dirname "$0")"
RUNS=${1:-1}
RESULTS=matrix_results.txt
echo "config,runs,faults,completed,other,last_iter_seen" > "$RESULTS"

run_config() {
  local label="$1"; shift
  local faults=0 completed=0 other=0 last_iter=""
  for r in $(seq 1 "$RUNS"); do
    local log="logs/matrix_${label}_$r.log"
    if [ "$label" = "sparse_launch_blocking" ]; then
      CUDA_LAUNCH_BLOCKING=1 python3 -u reproducer.py "$@" > "$log" 2>&1
    else
      python3 -u reproducer.py "$@" > "$log" 2>&1
    fi
    if grep -q "Memory access fault" "$log"; then
      faults=$((faults + 1))
    elif grep -q "COMPLETED" "$log"; then
      completed=$((completed + 1))
    else
      other=$((other + 1))
    fi
    last_iter=$(grep -oE "iter +[0-9]+" "$log" | tail -1)
  done
  echo "$label,$RUNS,$faults,$completed,$other,$last_iter" >> "$RESULTS"
  echo "$label: faults=$faults completed=$completed other=$other (last: $last_iter)"
}

# baseline and dtype
run_config baseline_sparse_bf16 --impl sparse --dtype bf16
run_config sparse_fp32 --impl sparse --dtype fp32
# controls: disable one component at a time
run_config sparse_no_moe --impl sparse --first_k_dense 6
run_config sparse_no_mtp --impl sparse --n_mtp 0
run_config sparse_no_eval --impl sparse --no_eval
run_config sparse_no_optimizer --impl sparse --no_optimizer
run_config sparse_launch_blocking --impl sparse
# shape / routing variations
run_config sparse_batch16 --impl sparse --batch_size 16
run_config sparse_block64 --impl sparse --block_size 64 --sliding_window 32
run_config sparse_experts4 --impl sparse --n_experts 4
run_config sparse_experts16 --impl sparse --n_experts 16
run_config sparse_all_experts --impl sparse --n_experts_per_tok 8
# alternative implementations / devices (fault boundary)
run_config dense_bf16 --impl dense --dtype bf16
run_config cpu_control --impl sparse --device cpu --n_layer 2 --n_embd 128 --batch_size 16 --block_size 64 --iters 20
# minimal-size attempt
run_config sparse_small --impl sparse --n_layer 2 --n_embd 128 --batch_size 16 --block_size 64 --sliding_window 32 --iters 500

echo "=== matrix done ===" >> "$RESULTS"
cat "$RESULTS"
