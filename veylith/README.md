# veylith

A from-scratch language model family, trained on Tiny Shakespeare on a
single AMD RX 9070 XT (ROCm, bf16). Every weight starts random and is
learned by gradient descent on this machine. No pretrained anything.

## Project status

veylith is a research project, not a product. v1.x and v2.x are
experimental research prototypes and feasibility studies. They are not
finished, production-ready or generally usable language models, and no
checkpoint in this repository is a released model.

The experiments study transformer and MoE architectures, training and
inference implementations, ROCm/GPU compatibility, checkpointing and
resume, routing, attention and multi-token prediction, overfitting and
generalization, and reproducible evaluation.

| label | meaning |
|---|---|
| v1.x | Experimental Transformer Baselines |
| v2.x | Experimental MoE Architecture Research |
| v3.0 | Pretraining Data Infrastructure |
| v3.1 | Initial Pretraining Experiments |

Software version, architecture version, training run, checkpoint and
published model are separate things and must be named separately. The tag
`v2-research-freeze-2026-10-26` marks a frozen research state, not a model
release.

## Layout

- `driver/` contains the native Rust expert-3 evaluator, E16/K4 CPU training check and plain dense Transformer trainer. Run Cargo from the repository root. [Rust commands and protocols](driver/README.md)
- `v1/` and `v2/` preserve the baseline and MoE checkpoints, tokenizer files and research records.
- `v3/` contains pretraining and routing evidence, including the frozen numerical references. [v3 research](v3/README.md)
- `reference/` - `config.json` files of DeepSeek-V4-Pro, GLM-5,
  Muse Glimmer 30B and MiMo-V2.5. Ground truth for every design choice
  in v2. Grep them: `grep num_experts reference/*.json`.

## Results so far

| model | tokenizer | params | val loss |
|---|---|---|---|
| v1 | char (vocab 65) | 4.84M | 1.4749 nats/token (plain next-token CE) |
| v1 | BPE (vocab 1000) | 5.32M | 3.2903 nats/token (plain next-token CE) |
| v2 | BPE (vocab 1000) | 25.60M | 6.0358 composite (training objective) |

v2's composite is `main_ce + 0.5 * mtp_ce + 0.01 * aux_loss`; it is the
training objective, not a quality metric. On a fixed 16-window validation
set the components are main 3.1895, MTP 4.9480 and aux 5.0889 (composite
5.7144). The stored 6.0358 comes from the training run's random-window
evaluations, so it is not the sum of those components. See
`v2/README.md` for the definitions and `v2/STATUS-2026-10-26.md` for the
measurement.

Per-character conversion, for reference only:
`bits/char = nats/token / (chars/token * ln 2)` with
`chars/token = 1,115,394 / 492,841 = 2.2632` for this corpus, and one
character per token for the char model. That gives 2.128 (v1 char),
2.097 (v1 BPE) and 2.033 (v2 main CE). The previously published 2.181 (v1)
could not be reconstructed. The v2 value 2.256 matches a training-time main
CE of 3.539 nats/token reported in commit 347aa70 (random windows), not the
3.18 measured here on fixed sections. Both old numbers are withdrawn as
comparison values; they are not comparable to the values above.

Do not compare rows directly: v1 reports the plain next-token CE, v2
reports a composite that includes the MTP term, and tokenizer, corpus and
evaluation procedure also differ.

A controlled evaluation on six shared validation sections (identical text,
main next-token CE only) gives 1.913 bits/char for v1 char, 1.922 for v1
BPE, 2.060 for the shipped v2 BPE checkpoint, 2.046 for the v2 checkpoint
from the controlled 5000-step run (selected at step 500 by main CE) and
3.025 for the 120-step diagnostic checkpoint. On the same BPE tokens the
controlled v2 is 0.189 nats/token worse than v1 (paired over sections,
SE 0.034); against the shipped v2 it is 0.024 +/- 0.026 nats/token better,
inside one standard error. Protocol, router analysis and generation
examples: `evaluation/REPORT.md` and `evaluation/P1-3-REPORT.md`; raw
numbers in `evaluation/results.json` and `evaluation/p1-3-results.json`.

From the original training run (log not stored in this repo), v2's
validation loss bottomed around iter 500 (6.0358) and the final evaluation
at iter 3000 was 8.06 while the training loss was 1.39: 25.6M params on
0.5M tokens is too much capacity for the data, and early stopping kept the
best weights. The controlled run reproduces that shape on fixed sections:
selection CE bottoms at step 500 (3.278) and rises to 4.75 at step 4999
while training CE falls to 0.59. A check of the saved checkpoint on three
validation batches gives a mean router probability between 0.080 and 0.208
per expert (target 0.125), with no collapsed expert. The lesson: capacity
must match data. A fancier architecture does not beat a smaller model when
the corpus is tiny.

### v3.1 expert 3 routing

At update 500, replacing expert 3 at the selected punctuation positions raised next-token loss by **0.186 nats** on the original sample and **0.231 nats** on confirmation. Rust and Python on the Radeon RX 9070 XT reproduced the CPU replacement counts of **4,180** and **4,299** and gave identical GPU loss estimates and confidence intervals. [Expert 3 study results](v3/docs/ROUTING-STUDY.md#expert-3-results-at-update-500)

Replacing expert 3 also raised loss at matched word pieces. The punctuation-minus-word difference was **−0.085 nats** on the original sample and **+0.010** on confirmation. Both paired 95% intervals included zero. [Matched comparisons and coverage](v3/docs/ROUTING-STUDY.md#expert-3-at-matched-punctuation-and-word-pieces)

Across eight GPU runs, mean wall time through verified output and process exit was **64.0 s for Rust** and **76.9 s for Python**. The earlier Rust CPU mean was **327.9 s**. [Timings by validation sample](../README.md#expert-3-benchmark) · [GPU benchmark data](driver/expert3-rocm-benchmark.json)

### Rust training comparisons

A synthetic 16-expert, top-4 CPU training step matched the archived Python router on every selection. Both Rust implementations reproduced the training loss exactly, and every AdamW parameter change agreed within 2.99 × 10⁻⁸. [E16/K4 measurements](v3/docs/E16-K4-CPU-PARITY.md)

The 1.13M-parameter plain dense Transformer trained on the same 16,352,000 targets as v3.1-L. It reached a source-macro validation loss of **6.90331**. The 98.34M-parameter MoE reached **5.40009** under the same token budget. Both scores use the complete validation split. [Dense training results, controls and checkpoint hashes](v3/docs/DENSE-BASELINE-PROTOCOL.md#results-on-19-november-2026)

## Reproduce historical Python runs

Switch to `archive/python-research` for the original training, corpus preparation, checkpoint sampling and evaluation commands. Return to `main` for the Rust evaluator. [Archive quickstart](https://github.com/VikingOwl91/VeylithAI/tree/archive/python-research)

The v2 baseline is frozen (tag `v2-research-freeze-2026-10-26`):
architecture, protocols, results, limitations and checkpoint hashes are
recorded in `v2/FREEZE-2026-10-26.md`. The original v3 data work is preserved in the Python archive.

v3.0 is data infrastructure, not a model. The first pilot materialized
2000 FineWeb-Edu documents (1.92M tokens, eight shards, 8192-token BPE) on
CPU, with a manifest digest, shard hashes and deep verification. Numbers,
limitations and the 10M-token plan are in `v3/STATUS-2026-10-26.md`.

## License and training data

You may redistribute and modify Veylith's original software under the GNU General Public License, version 3 or (at your option) any later version (`GPL-3.0-or-later`). The full terms are in [LICENSE](../LICENSE).

Training text, tokenized copies and third-party reference configurations retain their respective upstream terms. The v1 and v2 checkpoints are experimental artifacts. This project grants no separate license for the checkpoint weights or trained tokenizer models at this time. See [training data origins and licenses](TRAINING-DATA.md) for dataset attribution, tracked copies and the sources used by each generation.
