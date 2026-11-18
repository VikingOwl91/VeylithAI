#!/usr/bin/env bash
# Paired benchmark of the frozen Python-backed driver and the native Rust driver.
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: $0 LEGACY_DRIVER NATIVE_DRIVER DATA_ROOT CHECKPOINT NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
legacy=$(realpath -e -- "$1")
native=$(realpath -e -- "$2")
data=$(realpath -e -- "$3")
checkpoint=$(realpath -e -- "$4")
v3=$(cd -- "$(dirname -- "$0")/../v3" && pwd)
out=$(realpath -m -- "$5")
cpus=${VEYLITH_BENCH_CPUS:-4,5}
mkdir -- "$out"
mkdir -- "$out/tmp" "$out/native-path"
ln -s -- "$(command -v bsdtar)" "$out/native-path/bsdtar"
export TMPDIR="$out/tmp" LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
TIMEFORMAT='%3R'
index=0

run() {
    local sample=$1 backend=$2
    local -a command
    index=$((index+1))
    local prefix="$out/$index-$sample-$backend"
    case "$backend" in
        rust) command=(env "PATH=$out/native-path" "$native") ;;
        python) command=("$legacy" --v3-dir "$v3") ;;
    esac
    echo "Run $index/8: $sample, $backend" >&2
    if ! { time taskset -c "$cpus" "${command[@]}" --sample "$sample" --data-root "$data" \
        --checkpoint "$checkpoint" --out "$prefix.report.json" \
        > "$prefix.receipt.json" 2> "$prefix.log"; } 2> "$prefix.wall-seconds"; then
        tail -n 5 -- "$prefix.log" >&2
        return 1
    fi
    jq -e --arg sample "$sample" '.verified == true and .sample == $sample' "$prefix.receipt.json" >/dev/null
    echo "Verified in $(cat -- "$prefix.wall-seconds") s wall time." >&2
}

run original rust
run original python
run confirmation python
run confirmation rust
run original python
run original rust
run confirmation rust
run confirmation python
