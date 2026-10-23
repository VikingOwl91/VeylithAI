#!/usr/bin/env bash
# Multi-seed layer bisect in experts-only mode (defeats the RNG-shift confound).
set -u
cd "$(dirname "$0")"
RESULTS=layer_bisect_results.txt
echo "n_layer,seed,result,last_iter" > "$RESULTS"
for nl in 2 3 4 5 6; do
  for seed in 1337 42 7; do
    log="logs/lb_L${nl}_s${seed}.log"
    python3 -u reproducer.py --no_optimizer --grad_filter experts --n_layer $nl --seed $seed --iters 100 > "$log" 2>&1
    if grep -q "Memory access fault" "$log"; then r=FAULT
    elif grep -q COMPLETED "$log"; then r=COMPLETED
    else r=OTHER; fi
    last=$(grep -oE "iter +[0-9]+" "$log" | tail -1)
    echo "L$nl,s$seed: $r ($last)"
    echo "$nl,$seed,$r,$last" >> "$RESULTS"
  done
done
echo "=== layer bisect done ===" >> "$RESULTS"
cat "$RESULTS"
