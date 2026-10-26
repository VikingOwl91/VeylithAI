# veylith

A from-scratch language model family, trained on Tiny Shakespeare on a
single AMD RX 9070 XT (ROCm, bf16). Every weight starts random and is
learned by gradient descent on this machine. No pretrained anything.

## Layout

- `v1/` - the original: char-level or custom-BPE tokenizer, learned
  position embeddings, LayerNorm, full multi-head attention, GELU MLP.
  Frozen, kept exactly as it was.
- `v2/` - the upgrade: RoPE, RMSNorm, GQA, sliding-window attention,
  SwiGLU, Mixture-of-Experts, multi-token prediction. A miniature of a
  2026 frontier architecture, still trained from scratch in minutes.
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
BPE, 2.060 for v2 BPE and 3.025 for the 120-step diagnostic checkpoint. On
the same BPE tokens v2 is 0.213 nats/token worse than v1 (paired over
sections, SE 0.038). Protocol, router analysis and generation examples:
`evaluation/REPORT.md`; raw numbers in `evaluation/results.json`.

From the original training run (log not stored in this repo), v2's
validation loss bottomed around iter 500 (6.0358) and the final evaluation
at iter 3000 was 8.06 while the training loss was 1.39: 25.6M params on
0.5M tokens is too much capacity for the data, and early stopping kept the
best weights. A check of the saved checkpoint on three validation batches
gives a mean router probability between 0.080 and 0.208 per expert
(target 0.125), with no collapsed expert. The lesson: capacity must match
data. A fancier architecture does not beat a smaller model when the corpus
is tiny.

## Run

```bash
cd v1 && python train.py            # v1, char tokenizer
cd v1 && python train.py --tokenizer bpe --bpe_vocab_size 1000
cd v2 && python train.py --tokenizer bpe --bpe_vocab_size 1000
```

Each version is self-contained: run from inside its folder, paths are
relative, the dataset downloads itself on first run.

Evaluate every checkpoint on the shared CPU protocol (no GPU, writes
`evaluation/results.json`):

```bash
python3 veylith/evaluate.py
```
