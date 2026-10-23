#!/usr/bin/env bash
# Backward bisect: which parameter group's backward is necessary for the fault?
set -u
cd "$(dirname "$0")"
RESULTS=grad_bisect_results.txt
echo "config,result,last_iter" > "$RESULTS"
run_config() {
  local label="$1"; shift
  local log="logs/gb_${label}.log"
  python3 -u reproducer.py --no_optimizer --iters 100 "$@" > "$log" 2>&1
  if grep -q "Memory access fault" "$log"; then r=FAULT
  elif grep -q COMPLETED "$log"; then r=COMPLETED
  else r=OTHER; fi
  local last=$(grep -oE "iter +[0-9]+" "$log" | tail -1)
  echo "$label,$r,$last" >> "$RESULTS"
  echo "$label: $r ($last)"
}
run_config gf_all
run_config gf_experts --grad_filter experts
run_config gf_gate --grad_filter gate
run_config gf_shared --grad_filter shared
run_config gf_attn --grad_filter attn
run_config gf_emb_head --grad_filter emb_head
run_config gf_ln --grad_filter ln
run_config gf_moe_params --grad_filter experts,gate,shared
run_config gf_non_moe --grad_filter attn,emb_head,ln
echo "=== grad bisect done ===" >> "$RESULTS"
cat "$RESULTS"
