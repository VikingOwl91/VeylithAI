# Veylith expert 3 evaluator

The Rust driver evaluates the frozen update-500 checkpoint on CPU or an AMD GPU using the system LibTorch library. It reads the training checkpoint and token shards, runs ordinary and intervened inference, computes next-token losses and source-stratified bootstrap intervals, and publishes a verified report. The archived Python evaluator and its two reports are the numerical reference. [Experiment and scientific results](../v3/docs/ROUTING-STUDY.md#expert-3-results-at-update-500)

## Run a sample

Build on Linux with Rust, a C++ compiler, LibTorch 2.15.0 headers and libraries, and `bsdtar` on PATH. The tested build used GCC with the system LibTorch installation. Set `DATA` to the frozen multi-100m corpus root, `CKPT` to the original update-500 training checkpoint, and `OUT` to a **new** report path in an existing directory. Allow roughly 1.5 GiB of temporary disk space alongside the report. The loader normalizes the checkpoint ZIP into a temporary file and removes it after reading the verified model tensors.

From the Git repository root:

```sh
export CARGO_TARGET_DIR="$PWD/.codex/target" CXX=g++
cargo build -j 2 --release --locked --manifest-path veylith/driver/Cargo.toml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$CARGO_TARGET_DIR/release/veylith-driver" --sample original \
  --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

Select `--sample confirmation` and a fresh output path for the second sample. The default evaluator uses CPU float32, two LibTorch compute threads and one interop thread. It inherits the caller’s CPU affinity and lowers its scheduling priority to niceness 15 if needed. To pin the run to two particular logical CPUs on Linux, launch with `taskset -c`. The old driver’s `--python` and `--v3-dir` flags applied to its Python worker; the native CLI takes its complete inputs from the two artifact paths.

## What gets checked

* Checkpoint, corpus manifest, tokenizer, consumed validation shards, ordered tokens and the 229 loaded model parameter payloads are checked against recorded SHA-256 digests. The second sample is checked for sequence overlap with the first.
* The forward pass shares computation up to the block-1 intervention. It evaluates each block-1 expert once on the common input and combines its output separately with the ordinary and intervened mixture weights. Later blocks run independently. Each substitute receives expert 3’s original weight.
* The evaluator checks loss at every position before its sequence’s first intervention, then compares all five outcome groups including counts, CE, deltas and 95% intervals to the corresponding frozen report. CPU runs require an exact match. GPU runs use the numerical tolerance described below. The Rust bootstrap reproduces all 240,000 NumPy row selections for this protocol’s fixed seed.
* The schema-2 report identifies the compiled Rust sources and dependencies. Its Git fields describe the driver package at build time. The evaluator stages the verified report on the destination filesystem and publishes it without replacing an existing output.

Progress goes to stderr. The JSON receipt on stdout measures wall time from driver entry through publication. It also includes time spent verifying inputs, loading weights and batches, evaluating token losses, and computing bootstrap statistics.

## Benchmark the change

The benchmark compares this release binary against the Rust driver from commit `e97f44b` that launches the frozen Python evaluator. Build that reference separately so both binaries are available. Use the same Python evaluation environment as the frozen reports. Pass the two binaries, original input paths and a new output directory to the benchmark script.

```sh
LEGACY_BUILD="$PWD/.codex/expert3-python-reference"
mkdir -p "$LEGACY_BUILD"
git archive e97f44b -- veylith/driver veylith/v3/docs/routing-expert3-update500.json \
  veylith/v3/docs/routing-expert3-update500-confirmation.json | tar -x -C "$LEGACY_BUILD"
cargo build -j 2 --release --locked --target-dir "$LEGACY_BUILD/target" \
  --manifest-path "$LEGACY_BUILD/veylith/driver/Cargo.toml"
LEGACY_DRIVER="$LEGACY_BUILD/target/release/veylith-driver"
NATIVE_DRIVER="$CARGO_TARGET_DIR/release/veylith-driver"
veylith/driver/benchmark-expert3.sh "$LEGACY_DRIVER" "$NATIVE_DRIVER" "$DATA" "$CKPT" "$NEW_BENCH_DIR"
```

The script runs both binaries twice per sample, reversing their order on the second pair. It pins all eight processes to the same logical CPUs (4 and 5 by default, configurable with `VEYLITH_BENCH_CPUS`). Each run includes input checks, fresh model loading, inference, statistics and report publication. Native runs receive a PATH containing only `bsdtar`. Each output includes the report, driver receipt, progress log, and independently timed process wall duration. The two older single-run CPU measurements are retained for context. [Historical Python-backed driver receipts](expert3-cpu-timings.json)

## Paired CPU results

Measured on an AMD Ryzen 7 3700X on 18 November 2026. Every run reproduced its frozen report's scientific fields exactly. The table gives whole-process wall seconds, including startup and exit. Each row compares the same checkpoint and ordered token sample.

| Sample | Pair | First to run | Python (s) | Rust (s) |
|---|---:|---|---:|---:|
| Original | 1 | Rust | 553.102 | 356.855 |
| Original | 2 | Python | 426.950 | 317.042 |
| Confirmation | 1 | Python | 531.601 | 307.756 |
| Confirmation | 2 | Rust | 499.183 | 330.058 |

Across four runs per implementation, mean wall time fell from **502.7 s to 327.9 s**. The native evaluator saved **174.8 s per run (34.8%)** and finished first in each pair. This measures the two implementations as shipped, including the Rust evaluator's reuse of shared model computation. The individual timings show the variation on the shared workstation. [All eight receipts, build revisions and report hashes](expert3-native-benchmark.json)

## Run the ROCm comparison

The native GPU evaluator requires the ROCm build of LibTorch and HIP development headers. Enable the Cargo feature and select the GPU at runtime. The driver obtains HIP’s lowest supported stream priority, verifies the actual priority, moves the model to device 0, and copies each batch’s losses and routing decisions back to the CPU for reduction. The measured configuration set an eight-bit compute-unit mask and ran one batch at a time. [AMD ROCm compute-unit settings](https://rocm.docs.amd.com/en/docs-7.2.1/reference/env-variables.html)

Run this from the repository root with the same frozen inputs and a new output path.

```sh
export CARGO_TARGET_DIR="$PWD/.codex/target" CXX=g++ LIBTORCH=/usr ROCM_PATH=/opt/rocm
cargo build -j 2 --release --locked --features rocm --manifest-path veylith/driver/Cargo.toml
export ROCR_VISIBLE_DEVICES=0 ROC_GLOBAL_CU_MASK=0xff
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 15 taskset -c 4,5 "$CARGO_TARGET_DIR/release/veylith-driver" --device rocm \
  --sample original --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

The Python comparator uses the archived evaluator with a small patch for GPU inference, stream selection, and verification. It runs both full forward passes and uses the existing CPU reduction and bootstrap code. These commands prepare a fresh reference directory and run the original sample.

```sh
GPU_REFERENCE="$PWD/.codex/expert3-rocm-reference"
mkdir "$GPU_REFERENCE"
git archive 626772b -- veylith/v3 | tar -x -C "$GPU_REFERENCE"
patch -d "$GPU_REFERENCE" -p1 < veylith/driver/expert3-python-rocm.patch
nice -n 15 taskset -c 4,5 python -B "$GPU_REFERENCE/veylith/v3/analysis/expert3_study.py" \
  --data-root "$DATA" --checkpoint "$CKPT" --threads 2 --out "$PYTHON_OUT"
```

For confirmation use `--sample confirmation` in Rust and `--fresh-sample` in Python. Use a fresh output path each time. Both comparators check the frozen input hashes, exact counts, and exact equality of the two forward passes at positions preceding the first replacement. Against the CPU reference, the absolute tolerance is **0.0001 nats** for each baseline mean, mean loss change, and endpoint of its 95% interval.

## Paired GPU results

Both evaluators ran on the workstation’s Radeon RX 9070 XT using HIP stream priority +1 and the `0xff` compute mask. Each value is whole-process wall time through verified output and exit. Fresh processes shared the workstation’s file cache and the same ROCm kernel cache.

| Validation sample | Pair | First to run | Python GPU (s) | Rust GPU (s) |
|---|---:|---|---:|---:|
| Original | 1 | Rust | 72.870 | 64.256 |
| Original | 2 | Python | 86.384 | 67.179 |
| Confirmation | 1 | Python | 74.471 | 62.052 |
| Confirmation | 2 | Rust | 73.831 | 62.525 |

Rust averaged **64.0 s** and Python **76.9 s**, a **16.8%** reduction in GPU wall time. The previous Rust CPU mean was 327.9 s, or **5.12×** the GPU time. The GPU Python comparator runs the evaluation script directly. The historical CPU Python benchmark also timed its Rust launcher’s preflight checks.

All eight runs passed. Within each sample, Python and Rust on GPU produced identical reported counts, CE values and intervals. Against the frozen CPU reports, substitution counts matched exactly at 4,180 and 4,299. The largest absolute difference across all CE values and interval endpoints was **8.42e-07 nats**.

The lowest sampled free VRAM was **7.83 GiB**. The Wayland probe on one display measured a **10.009 ms** 95th-percentile frame interval before the benchmark and **10.009 ms** in each run. The longest interval during the runs was **30.018 ms**, compared with 10.011 ms during the 12-second preflight. These timings describe the probe surface. Application input latency was unmeasured. [Per-run timings, resource measurements and scientific results](expert3-rocm-benchmark.json)
