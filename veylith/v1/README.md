> Part of the veylith family. See `../README.md` for the family overview.

# veylith v1: train your own language model from scratch

The Python commands in this historical guide run on `archive/python-research`. [Browse the archived directory](https://github.com/VikingOwl91/VeylithAI/tree/archive/python-research/veylith/v1)

A single-file, dependency-free (beyond PyTorch) implementation of a small
GPT-style transformer, trained on Tiny Shakespeare. Two tokenizer options:
character-level (simplest) or a custom subword BPE you train yourself.
No pretrained weights anywhere: every parameter starts random and is
learned by gradient descent on your machine.

## Requirements

- Python 3.10+
- PyTorch with GPU support:
  - AMD: ROCm build (`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`)
  - NVIDIA: CUDA build (default `pip install torch`)
  - CPU-only works too, just slow
- `sentencepiece` (only needed for `--tokenizer bpe`)

## Run

```bash
python train.py                                           # char tokenizer
python train.py --tokenizer bpe --bpe_vocab_size 1000     # custom BPE tokenizer
python sample.py                                          # load checkpoint, generate
```

Quick smoke test (about a minute, tiny model):

```bash
python train.py --n_layer 2 --n_embd 64 --n_head 2 --batch_size 32 --block_size 64 --max_iters 60 --eval_interval 30
```

## The custom BPE tokenizer

`--tokenizer bpe` trains a subword BPE tokenizer on the corpus with
sentencepiece, the "mini-Gemma" recipe: BPE merges plus `byte_fallback`,
so any unseen character decomposes into byte tokens and nothing is ever
unknown. It is trained once and cached in `data/`
(`tinyshakespeare_bpe1000.model` plus the tokenized ids), then reused.

Vocab size is the knob: a bigger vocab means shorter sequences (less
compute per character, more context per window) but a bigger embedding
matrix. 1000 is a good start for Shakespeare.

Comparing runs across tokenizers: per-token loss is NOT comparable
(char-level starts near ln(65) = 4.18, BPE near ln(vocab_size)). Compare
bits per character or sample quality instead.

## What you should see

Loss starts near ln(vocab_size) (4.18 for char-level) and should fall
well below 2. Generated text goes from keyboard soup to Shakespeare-ish.

## Scale it up

| knob | default | bigger |
|---|---|---|
| `--n_layer` | 6 | 8-12 |
| `--n_embd` | 256 | 384-512 |
| `--block_size` | 256 | 512 |
| `--batch_size` | 64 | 128 |
| `--max_iters` | 5000 | 20000+ |
| `--bpe_vocab_size` | 1000 | 2000-8000 |

Swap the dataset by changing `DATA_URL` in `train.py` (e.g. TinyStories
for clean English stories).

## Notes

- On AMD GPUs (ROCm), PyTorch exposes the GPU as the `cuda` device, so
  `pick_device()` needs no AMD-specific code. bf16 mixed precision is
  used automatically when supported.
- Everything is seeded (`--seed`), so runs are reproducible.
- The saved checkpoint holds the best-validation weights (early stopping),
  not the final ones. If val loss starts rising while train loss falls,
  the model is memorizing; the saved weights come from before that.
- The model, attention, and training loop are written out by hand on
  purpose. Read `train.py` top to bottom; it is the whole recipe.
