# Veylith — Python research archive

Veylith is a family of small language models trained from scratch to study a practical question: when does routing tokens to different experts improve prediction? The project started with Transformer baselines on Tiny Shakespeare and grew into multi-source pretraining experiments on a single AMD Radeon RX 9070 XT. The repository includes PyTorch implementations, experimental checkpoints, and the measurements behind the results.

## Using this branch

`archive/python-research` preserves the Python training, data preparation, checkpoint sampling and evaluation workflows. Its Python source is frozen at commit `9acb457`. Use this branch to reproduce the original expert 3 reports or to work with the historical models. Continued Rust development lives on [main](https://github.com/VikingOwl91/VeylithAI/tree/main).

From the repository root, set `DATA` to the frozen multi-100m corpus root, `CKPT` to the update-500 checkpoint and `OUT` to a new report path. In the study’s Python environment, run the CPU reference with two threads.

```sh
python -B veylith/v3/analysis/expert3_study.py --threads 2 \
  --data-root "$DATA" --checkpoint "$CKPT" --out "$OUT"
```

Add `--fresh-sample` and choose a different output path for confirmation. The ROCm comparison used the same Python source with the [GPU adapter patch](veylith/driver/expert3-python-rocm.patch). The [benchmark snapshot](https://github.com/VikingOwl91/VeylithAI/blob/225d574840820ae71f388651a9ad07bd45290b64/veylith/driver/README.md) preserves the exact paired CPU and GPU commands and measurements.

## Try a checkpoint

From the repository root, run the following in a Python environment. This generates a short passage using the included v2 checkpoint and tokenizer. A CPU is sufficient.

```sh
cd veylith/v2
python -m pip install -r requirements.txt
python sample.py --prompt "JULIET:" --max_new_tokens 40 --seed 42
```

## What the routing experiment found

At training update 500, the v3.1 model already routed numbers, punctuation, and end-of-document tokens to particular experts. To test whether those choices helped prediction, we shuffled the expert assignments and their weights together in one layer at a time. Each expert kept the same total load within the batch. Only the pairing between tokens and routing decisions changed at that layer. [Expert behavior](veylith/v3/docs/V3.1-N-EXPERT-BEHAVIOR.md) · [Experiment design](veylith/v3/docs/ROUTING-STUDY.md)

On 240 validation sequences, the unmodified model scored **5.44047 nats per predicted token**. Shuffling increased loss in all seven mixture-of-experts layers when averaged over three seeds. The largest increase was **0.09682** in the first expert layer and the smallest was **0.03084** in the last. This is evidence that the learned routing choices helped this checkpoint predict the next token on this sample. [Measurements and uncertainty intervals](veylith/v3/docs/routing-study-update500.json)

## Explore the project

| Area | What you can study |
|---|---|
| [v1 · Transformer baselines](veylith/v1/README.md) | Compare character and byte-pair tokenization on the same plays. |
| [v2 · Mixture of experts](veylith/v2/README.md) | Train, sample, and resume a model with grouped-query attention, top-2 routing, and multi-token prediction. |
| [v1/v2 evaluation](veylith/evaluation/REPORT.md) | Compare checkpoints on the same held-out text using bits per character across tokenizers. |
| [v3 · Data and pretraining](veylith/v3/README.md) | Follow the corpus from pinned sources through tokenization and validation splits to a [500-update training run](veylith/v3/docs/V3.1-L-DENSE-RUN.md). |

The v3.1 routing evaluator uses a separately stored checkpoint and corpus. The experiment guide identifies both inputs, and the committed measurement file can be read independently.

## License and data

Veylith's original software is licensed under [GPL-3.0-or-later](LICENSE). Dataset terms and attribution are recorded for Tiny Shakespeare, FineWeb-Edu, English Wikipedia, and Cosmopedia v2. The project currently grants no separate license for the experimental checkpoint weights or trained tokenizer models. [Training data origins and licenses](veylith/TRAINING-DATA.md)
