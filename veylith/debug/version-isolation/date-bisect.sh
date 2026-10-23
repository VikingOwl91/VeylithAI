#!/usr/bin/env bash
# Date-based binary search over 2.15.0.dev ROCm nightlies for the
# sparse-MoE fault fix. Known boundary: 20260824 faults (testE),
# 20261005 passes (testD). Finds the smallest date interval with the
# available builds.
#
# Constraints honored: isolated uv venvs, existing venvs never replaced,
# each bisect venv deleted after testing (wheels stay in the uv cache),
# GPU compatibility checked before running, GPU faults recorded as test
# outcomes and distinguished from install failures.
set -u
cd "$(dirname "$0")/.."   # veylith/debug
PYBASE="${HOME}"/veylith-debug-venvs
LOGDIR=logs
RESULTS=version-isolation/date-bisect-results.txt
echo "date,faults,completed,last_iter,note" > "$RESULTS"

mapfile -t DATES < <(curl -sL --max-time 60 "https://download.pytorch.org/whl/nightly/rocm7.14/torch/" \
  | grep -oE 'torch-2\.15\.0\.dev[0-9]+[+]rocm7\.14-cp314-cp314-manylinux[^"<>]*\.whl' \
  | grep -oE 'dev[0-9]+' | sed 's/dev//' | sort -u \
  | awk '$1 >= 20260824 && $1 <= 20261005')
N=${#DATES[@]}
echo "available builds in window: $N (${DATES[0]} .. ${DATES[$((N-1))]})"

test_build() {
  local date=$1
  local venv="$PYBASE/bisect-$date"
  local py="$venv/bin/python"
  if [ ! -x "$py" ]; then
    uv venv "$venv" --python /usr/bin/python3 >/dev/null 2>&1
    if ! uv pip install --python "$py" "torch==2.15.0.dev$date" \
        --index-url https://download.pytorch.org/whl/nightly/rocm7.14 >/dev/null 2>&1; then
      echo "$date,install_fail,0,0,no-wheel-or-install-error" >> "$RESULTS"
      rm -rf "$venv"; return 2
    fi
  fi
  if ! "$py" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "$date,gpu_fail,0,0,cuda-unavailable" >> "$RESULTS"
    rm -rf "$venv"; return 2
  fi
  local faults=0 completed=0
  for i in 1 2; do
    "$py" reproducer.py --no_optimizer --grad_filter experts --iters 100 \
        > "$LOGDIR/db_${date}_$i.log" 2>&1
    if grep -q "Memory access fault\|HSA_STATUS_ERROR" "$LOGDIR/db_${date}_$i.log"; then
      faults=$((faults + 1))
    elif grep -q COMPLETED "$LOGDIR/db_${date}_$i.log"; then
      completed=$((completed + 1))
    fi
  done
  local last
  last=$(grep -oE "iter +[0-9]+" "$LOGDIR/db_${date}_2.log" | tail -1)
  echo "$date,$faults,$completed,$last" >> "$RESULTS"
  rm -rf "$venv"
  if [ "$faults" -gt 0 ] && [ "$completed" -eq 0 ]; then return 1
  elif [ "$completed" -gt 0 ] && [ "$faults" -eq 0 ]; then return 0
  else return 3; fi
}

LO=0; HI=$((N - 1))
while [ $((HI - LO)) -gt 1 ]; do
  MID=$(( (LO + HI) / 2 ))
  echo "=== testing ${DATES[$MID]} (idx $MID; known fail ${DATES[$LO]}, known pass ${DATES[$HI]}) ==="
  test_build "${DATES[$MID]}"
  rc=$?
  if [ $rc -eq 1 ]; then LO=$MID
  elif [ $rc -eq 0 ]; then HI=$MID
  elif [ $rc -eq 2 ]; then echo "build untestable; stopping search"; break
  else echo "mixed result at ${DATES[$MID]}; stopping"; break; fi
done
echo "RESULT: fix window = [${DATES[$LO]} faults .. ${DATES[$HI]} passes]" | tee -a "$RESULTS"
