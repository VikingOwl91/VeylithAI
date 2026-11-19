# Veylith expert 3 evaluator

The Rust driver evaluates the frozen update-500 checkpoint on CPU or an AMD GPU using the system LibTorch library. It reads the training checkpoint and token shards, runs ordinary and intervened inference, computes next-token losses and source-stratified bootstrap intervals, and publishes a verified report. The archived Python evaluator and its two reports are the numerical reference. [Experiment and scientific results](../v3/docs/ROUTING-STUDY.md#expert-3-results-at-update-500)

The Cargo package also includes a one-step CPU training comparison for 16 experts and a plain dense Transformer trainer. [E16/K4 parity protocol and command](../v3/docs/E16-K4-CPU-PARITY.md) · [Dense baseline and resume command](../v3/docs/DENSE-BASELINE-PROTOCOL.md)

## Run a sample

Build on Linux with Rust, a C++ compiler, LibTorch 2.15.0 headers and libraries, and `bsdtar` on PATH. The tested build used GCC with the system LibTorch installation. Set `DATA` to the frozen multi-100m corpus root, `CKPT` to the original update-500 training checkpoint, and `OUT` to a **new** report path in an existing directory. Allow roughly 1.5 GiB of temporary disk space alongside the report. The loader normalizes the checkpoint ZIP into a temporary file and removes it after reading the verified model tensors.

Build and run from the Git repository root.

```sh
export CARGO_TARGET_DIR="$PWD/.codex/target" LIBTORCH=/usr CXX=g++
cargo build -j 2 --release --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$CARGO_TARGET_DIR/release/veylith-driver" --sample original \
  --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

Select `--sample confirmation` and a fresh output path for the second sample. The default evaluator uses CPU float32, two LibTorch compute threads and one interop thread. It inherits the caller’s CPU affinity and lowers its scheduling priority to niceness 15 if needed. To pin the run to two particular logical CPUs on Linux, launch with `taskset -c`. The old driver’s `--python` and `--v3-dir` flags applied to its Python worker; the native CLI takes its complete inputs from the two artifact paths.

To compare expert 3 at matched punctuation and word pieces, add `--study matched --device cpu` to each sample command. Adding `--coverage-only` publishes the eligible and matched counts and balance before any losses are evaluated. A scored run checks the ordinary loss against the full frozen sample and checks that each replacement leaves all earlier predictions unchanged. [Matching rule, minimum coverage and uncertainty](../v3/docs/EXPERT3-MATCHED-PROTOCOL.md)

## What gets checked

* Checkpoint, corpus manifest, tokenizer, consumed validation shards, ordered tokens and the 229 loaded model parameter payloads are checked against recorded SHA-256 digests. The second sample is checked for sequence overlap with the first.
* The forward pass shares computation up to the block-1 intervention. It evaluates each block-1 expert once on the common input and combines its output separately with the ordinary and intervened mixture weights. Later blocks run independently. Each substitute receives expert 3’s original weight.
* The evaluator checks loss at every position before its sequence’s first intervention, then compares all five outcome groups including counts, CE, deltas and 95% intervals to the corresponding frozen report. CPU runs require an exact match. GPU runs use the numerical tolerance described below. The Rust bootstrap reproduces all 240,000 NumPy row selections for this protocol’s fixed seed.
* The schema-2 report identifies the compiled Rust sources and dependencies. Source-hash paths are relative to the repository root. The Git fields describe the driver package, workspace manifest and lockfile at build time. The evaluator stages the verified report on the destination filesystem and publishes it without replacing an existing output.

Progress goes to stderr. The JSON receipt on stdout measures wall time from driver entry through publication. It also includes time spent verifying inputs, loading weights and batches, evaluating token losses, and computing bootstrap statistics.

## Reproduce the Python comparisons

The original Python evaluator, its tests and the ROCm adapter live on `archive/python-research`. The paired benchmark instructions at commit `225d574` use the source layout and lockfile that produced the published measurements. Use that revision for a historical timing comparison. [Python research archive](https://github.com/VikingOwl91/VeylithAI/tree/archive/python-research) · [Benchmark commands at the measured milestone](https://github.com/VikingOwl91/VeylithAI/blob/225d574840820ae71f388651a9ad07bd45290b64/veylith/driver/README.md) · [Historical Python-backed driver receipts](expert3-cpu-timings.json)

## Paired CPU results

Measured on an AMD Ryzen 7 3700X on 18 November 2026. Every run reproduced its frozen report's scientific fields exactly. The table gives whole-process wall seconds, including startup and exit. Each row compares the same checkpoint and ordered token sample.

| Sample | Pair | First to run | Python (s) | Rust (s) |
|---|---:|---|---:|---:|
| Original | 1 | Rust | 553.102 | 356.855 |
| Original | 2 | Python | 426.950 | 317.042 |
| Confirmation | 1 | Python | 531.601 | 307.756 |
| Confirmation | 2 | Rust | 499.183 | 330.058 |

Across four runs per implementation, mean wall time fell from **502.7 s to 327.9 s**. The native evaluator saved **174.8 s per run (34.8%)** and finished first in each pair. This measures the two implementations as shipped, including the Rust evaluator's reuse of shared model computation. The individual timings show the variation on the shared workstation. [All eight receipts, build revisions and report hashes](expert3-native-benchmark.json)

## Run on ROCm

The native GPU evaluator requires the ROCm build of LibTorch and HIP development headers. Enable the Cargo feature and select the GPU at runtime. The driver obtains HIP’s lowest supported stream priority, verifies the actual priority, moves the model to device 0, and copies each batch’s losses and routing decisions back to the CPU for reduction. The measured configuration set an eight-bit compute-unit mask and ran one batch at a time. [AMD ROCm compute-unit settings](https://rocm.docs.amd.com/en/docs-7.2.1/reference/env-variables.html)

Run this from the repository root with the same frozen inputs and a new output path.

```sh
export CARGO_TARGET_DIR="$PWD/.codex/target" CXX=g++ LIBTORCH=/usr ROCM_PATH=/opt/rocm
cargo build -j 2 --release --locked --features rocm
export ROCR_VISIBLE_DEVICES=0 ROC_GLOBAL_CU_MASK=0xff
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 15 taskset -c 4,5 "$CARGO_TARGET_DIR/release/veylith-driver" --device rocm \
  --sample original --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

For confirmation use `--sample confirmation`. Use a fresh output path each time. Both comparators check the frozen input hashes, exact counts, and exact equality of the two forward passes at positions preceding the first replacement. Against the CPU reference, the absolute tolerance is **0.0001 nats** for each baseline mean, mean loss change, and endpoint of its 95% interval.

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
