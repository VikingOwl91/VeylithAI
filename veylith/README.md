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

| model | tokenizer | params | val loss | per char |
|---|---|---|---|---|
| v1 | char (vocab 65) | 4.84M | 1.4749 nats/char | 2.128 bits/char |
| v1 | BPE (vocab 1000) | 5.32M | 3.2903 nats/token | 2.181 bits/char |
| v2 | BPE (vocab 1000) | 25.60M | 6.0358 total | 2.256 bits/char |

Per-token loss is not comparable across tokenizers; per char is
(nats/char = nats/token / chars-per-token). For v2, per char uses the
main next-token CE only; the reported total also carries the MTP and
MoE-aux terms.

v2's val loss bottomed at iter 500 and rose after (train 1.39 vs val
8.06 at iter 3000): 25.6M params on 0.5M tokens is too much capacity
for the data, and early stopping kept the best weights. Router balance
verified (mean prob per expert 0.10-0.19 against a 0.125 target). The
lesson: capacity must match data. A fancier architecture does not beat
a smaller model when the corpus is tiny.

## Run

```bash
cd v1 && python train.py            # v1, char tokenizer
cd v1 && python train.py --tokenizer bpe --bpe_vocab_size 1000
cd v2 && python train.py --tokenizer bpe --bpe_vocab_size 1000
```

Each version is self-contained: run from inside its folder, paths are
relative, the dataset downloads itself on first run.
