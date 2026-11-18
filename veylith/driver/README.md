# Veylith experiment driver

The first Rust milestone (VEYLITH-20) reproduces the two CPU expert 3 runs frozen on local `main` at `9acb457`. Rust selects the sample, verifies input and evaluator hashes, sets the worker's resource budget, and publishes the report once it matches the corresponding reference. The existing PyTorch evaluator computes the predictions. [Experiment and results](../v3/docs/ROUTING-STUDY.md#expert-3-results-at-update-500)

## Reproduce a sample

Run from the Git repository root on Linux with Rust, GNU `nice`, and the Python evaluation environment available. Set `DATA` to the multi-100m corpus root, `CKPT` to the update-500 checkpoint, and `OUT` to a new JSON filename in an existing directory. Use the frozen CPU environment for exact numerical reproduction.

```sh
export CARGO_TARGET_DIR="$PWD/.codex/target"
cargo build --release --locked --manifest-path veylith/driver/Cargo.toml
"$CARGO_TARGET_DIR/release/veylith-driver" --sample original \
  --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

Choose `--sample confirmation` and a new output path for the second sample. Use `--python` to select an interpreter or `--v3-dir` to specify the evaluator directory when running elsewhere. The worker requests two PyTorch threads, one interop thread, one OpenBLAS thread and a niceness increment of 15. It inherits the caller’s CPU affinity.

## Reading the result

The output file is the evaluator's full report, including the checkpoint and ordered token hashes, substitute counts, baseline losses, intervention deltas, and confidence intervals. The driver checks those fields against the bundled reference. A differing result makes the command fail. Publication uses a hard link from a temporary directory on the output filesystem, which must support hard links. An existing output path is preserved.

The command writes progress to stderr and a JSON timing receipt to stdout. `total_seconds` measures wall time from driver entry through publication of the verified report. It includes input hashing, Python startup, inference, report checks and publication. The receipt also gives the three phase times. Build the executable before timing a run.

## First CPU measurements

| Sample | Input checks | Evaluator | Verify and publish | Total to result |
|---|---:|---:|---:|---:|
| Original | 1.372 s | 588.252 s | 0.0011 s | 589.625 s |
| Confirmation | 1.203 s | 521.089 s | 0.0002 s | 522.292 s |

The runs used the same two logical CPUs sequentially. All outcome fields matched the corresponding frozen report, including the differences in loss and their confidence intervals. These single runs measure turnaround for each sample. Measuring a speedup requires a paired timing comparison on the same sample. [Timing receipts and report hashes](expert3-cpu-timings.json)
