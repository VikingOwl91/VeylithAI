# Veylith

Veylith is a family of small language models trained from scratch to study a practical question: when does routing tokens to different experts improve prediction? The project started with Transformer baselines on Tiny Shakespeare and grew into multi-source pretraining experiments on a single AMD Radeon RX 9070 XT. The active branch, `main`, uses Rust to evaluate frozen routing decisions, check MoE training arithmetic and train a small dense Transformer. It includes the experimental checkpoints and the measurements behind the results.

## Run the Rust evaluator

From the repository root, use Cargo to build, test and run Veylith. The CPU evaluator uses Rust, a C++ compiler, LibTorch 2.15.0 and `bsdtar`. On the tested Linux system, LibTorch is installed under `/usr`.

```sh
export LIBTORCH=/usr CXX=g++
cargo test --locked -j 2
cargo run --release --locked -j 2 -- --help
```

To evaluate the original 240-sequence sample, set `DATA` to the frozen multi-100m corpus directory, `CKPT` to the matching update-500 training checkpoint and `OUT` to a new file in an existing directory. These inputs are stored separately. Use `--sample confirmation` to evaluate the second sample.

```sh
cargo run --release --locked -j 2 -- --sample original \
  --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

CPU is the default. On supported AMD systems, build with the `rocm` feature and select `--device rocm`. The driver checks the inputs and scientific results before publishing the report. [Inputs, verification and ROCm instructions](veylith/driver/README.md)

## Python research archive

Use `main` for daily development. The `archive/python-research` branch preserves the original Python evaluator, v1/v2 checkpoint sampling, training, corpus preparation and historical tests. To reproduce one of those workflows, run `git switch archive/python-research` from a clean checkout, then `git switch main` to return to Rust. The Rust workspace includes the expert 3 evaluator, a one-step E16/K4 CPU check and a plain dense Transformer trained on the same token budget as v3.1-L. Earlier training runs and their data licenses remain documented here. [Rust training comparisons](veylith/README.md#rust-training-comparisons) · [Python archive and quickstart](https://github.com/VikingOwl91/VeylithAI/tree/archive/python-research)

## What the routing experiment found

At training update 500, the v3.1 model already routed numbers, punctuation, and end-of-document tokens to particular experts. To test whether those choices helped prediction, we shuffled the expert assignments and their weights together in one layer at a time. Each expert kept the same total load within the batch. Only the pairing between tokens and routing decisions changed at that layer. [Expert behavior](veylith/v3/docs/V3.1-N-EXPERT-BEHAVIOR.md) · [Experiment design](veylith/v3/docs/ROUTING-STUDY.md)

On 240 validation sequences, the unmodified model scored **5.44047 nats per predicted token**. Shuffling increased loss in all seven mixture-of-experts layers when averaged over three seeds. The largest increase was **0.09682** in the first expert layer and the smallest was **0.03084** in the last. This is evidence that the learned routing choices helped this checkpoint predict the next token on this sample. [Measurements and uncertainty intervals](veylith/v3/docs/routing-study-update500.json)

## Expert 3 benchmark

At update 500, replacing expert 3 with the router’s highest-ranked unused expert increased next-token loss by **0.186** and **0.231 nats** on two validation samples at the replaced punctuation positions. The Rust evaluator reproduced the Python results exactly, including replacement counts (4,180 and 4,299), loss estimates and confidence intervals. [Intervention and results](veylith/v3/docs/ROUTING-STUDY.md#expert-3-results-at-update-500)

| Validation sample | Python CPU (s) | Rust CPU (s) | Python GPU (s) | Rust GPU (s) |
|---|---:|---:|---:|---:|
| Original | 490.0 | 336.9 | 79.6 | 65.7 |
| Confirmation | 515.4 | 318.9 | 74.2 | 62.3 |
| **Overall mean** | **502.7** | **327.9** | **76.9** | **64.0** |

Each sample row averages two runs per implementation on each device, with the order reversed for the second pair. Wall time includes startup, checkpoint loading, verification, publication and exit. On two Ryzen 7 3700X cores, Rust used **34.8% less time** than the Python-backed driver. The Rust evaluator reuses computation shared by the ordinary and intervened predictions. [Individual CPU timings](veylith/driver/README.md#paired-cpu-results) · [CPU benchmark data](veylith/driver/expert3-native-benchmark.json)

On the Radeon RX 9070 XT, Rust used **16.8% less time** than the Python GPU evaluator and **80.5% less time** than its earlier CPU run average. Both GPU implementations produced identical scientific results for each sample. [Paired GPU timings and desktop frame measurements](veylith/driver/README.md#paired-gpu-results) · [GPU benchmark data](veylith/driver/expert3-rocm-benchmark.json)

## Explore the project

| Area | What you can study |
|---|---|
| [v1 · Transformer baselines](veylith/v1/README.md) | Compare character and byte-pair tokenization on the same plays. |
| [v2 · Mixture of experts](veylith/v2/README.md) | Inspect the historical model with grouped-query attention, top-2 routing and multi-token prediction. |
| [v1/v2 evaluation](veylith/evaluation/REPORT.md) | Compare checkpoints on the same held-out text using bits per character across tokenizers. |
| [v3 · Data and pretraining](veylith/v3/README.md) | Follow the corpus from pinned sources through tokenization and validation splits to a [500-update training run](veylith/v3/docs/V3.1-L-DENSE-RUN.md). |

The v3.1 routing evaluator uses a separately stored checkpoint and corpus. The experiment guide identifies both inputs, and the committed measurement file can be read independently.

## License and data

Veylith's original software is licensed under [GPL-3.0-or-later](LICENSE). Dataset terms and attribution are recorded for Tiny Shakespeare, FineWeb-Edu, English Wikipedia, and Cosmopedia v2. The project currently grants no separate license for the experimental checkpoint weights or trained tokenizer models. [Training data origins and licenses](veylith/TRAINING-DATA.md)
