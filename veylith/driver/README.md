# Veylith expert 3 evaluator

The Rust driver evaluates the frozen update-500 checkpoint on CPU using the system LibTorch library. It reads the training checkpoint and token shards, runs ordinary and intervened inference, computes next-token losses and source-stratified bootstrap intervals, and publishes a verified report. The archived Python evaluator and its two reports are the numerical reference. [Experiment and scientific results](../v3/docs/ROUTING-STUDY.md#expert-3-results-at-update-500)

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

Select `--sample confirmation` and a fresh output path for the second sample. The evaluator uses CPU float32, two LibTorch compute threads and one interop thread. It inherits the caller’s CPU affinity and lowers its scheduling priority to niceness 15 if needed. To pin the run to two particular logical CPUs on Linux, launch with `taskset -c`. The old driver’s `--python` and `--v3-dir` flags applied to its Python worker; the native CLI takes its complete inputs from the two artifact paths.

## What gets checked

* Checkpoint, corpus manifest, tokenizer, consumed validation shards, ordered tokens and the 229 loaded model parameter payloads are checked against recorded SHA-256 digests. The second sample is checked for sequence overlap with the first.
* The forward pass shares computation up to the block-1 intervention. It evaluates each block-1 expert once on the common input and combines its output separately with the ordinary and intervened mixture weights. Later blocks run independently. Each substitute receives expert 3’s original weight.
* The evaluator checks loss at every position before its sequence’s first intervention, then compares all five outcome groups including counts, CE, deltas and 95% intervals to the corresponding frozen report. The Rust bootstrap reproduces all 240,000 NumPy row selections for this protocol’s fixed seed.
* The schema-2 report identifies the compiled Rust sources and dependencies. Its Git fields describe the driver package at build time. The evaluator stages the verified report on the destination filesystem and publishes it without replacing an existing output.

Progress goes to stderr. The JSON receipt on stdout measures wall time from driver entry through publication. It also includes time spent verifying inputs, loading weights and batches, evaluating token losses, and computing bootstrap statistics.

## Benchmark the change

The benchmark compares this release binary against the Rust driver from commit `e97f44b` that launches the frozen Python evaluator. Build that reference separately so both binaries are available. Use the same Python evaluation environment as the frozen reports. Pass the two binaries, original input paths and a new output directory to the benchmark script.

```sh
veylith/driver/benchmark-expert3.sh "$LEGACY_DRIVER" "$NATIVE_DRIVER" "$DATA" "$CKPT" "$NEW_BENCH_DIR"
```

The script runs both binaries twice per sample, reversing their order on the second pair. It pins all eight processes to the same logical CPUs (4 and 5 by default, configurable with `VEYLITH_BENCH_CPUS`). Each run includes input checks, fresh model loading, inference, statistics and report publication. Native runs receive a PATH containing only `bsdtar`. Each output includes the report, driver receipt, progress log, and independently timed process wall duration. The two older single-run CPU measurements are retained for context. [Historical Python-backed driver receipts](expert3-cpu-timings.json)
