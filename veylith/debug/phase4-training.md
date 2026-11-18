# Phase 4 / Phase C: experimental sparse-MoE training

Environment: isolated uv venv `phase4` (torch 2.16.0.dev20261006+rocm7.14,
HIP 7.14.60850, + sentencepiece; testC preserved untouched). RX 9070 XT.
Disposable experiment directories `phase4/sparse-run` and `phase4/dense-run`.

## Setup

- Implementations: the recovered sparse MoE (`sparse_moe_v2_train.py`,
  commit bfafa70 code) vs the current dense-combined v2 (`v2/train.py`).
- Identical: seed 1337 (same model init), same tokenizer (sentencepiece BPE
  vocab 1000 trained on the same corpus, deterministic), same data order,
  optimizer (AdamW 3e-4, warmup+cosine over 1500 iters), batch 64, block
  256, MTP x1, aux weight 0.01, bf16 autocast.
- Run: 1500 iterations each, eval every 250 (50 eval iters).

## Results

| metric | sparse | dense |
|---|---|---|
| trained iterations | 1500, no fault | 1500, no fault |
| wall-clock it/s | 5.0-5.4 | 5.5-5.6 |
| GPU memory | flat 0.4GB | flat 0.4GB |
| val loss @250 | 6.4771 | 6.4714 |
| val loss @1000 | 6.6687 | 6.6677 |
| val loss @1500 | 6.7765 | 6.7650 |
| train loss @1499 | 3.3815 (eval batch 2.7876) | 3.3975 (eval batch 2.7944) |

**Findings:**

1. **The sparse implementation trains stably on the fixed build**: 1500
   iterations, zero faults, flat memory, no NaN/Inf. On torch 2.14.1 the
   identical run faults within ~50 iterations.
2. **Numerical equivalence holds in real training**: the loss curves
   differ by ~0.01 (GPU nondeterminism) - consistent with Phase C's
   microbenchmark equivalence, on identical init/data/tokenizer.
3. **Wall-clock throughput is nearly equal at this batch size** (5.4 vs
   5.6 it/s) even though pure GPU step time favors sparse by 12%
   (Phase D): the wall-clock loop is bounded by CPU-side data loading
   (`get_batch`: randint + stack + H2D, ~50ms/iter), which masks the
   GPU-side advantage. Real-training throughput gains from sparse MoE
   require the data path to keep up (pre-staged batches, pinned memory,
   or a dataloader thread).
4. Both runs' `exit 1` was a harness bug (checkpoint path without a
   directory passed to `os.makedirs("")`), not a training failure; fixed
   and rerun for the checkpoint-based router analysis.

## Router load distribution (post-training, validation batches)

Both models converge to nearly identical, healthy routing distributions
(mean router probability per expert; target 0.125 for 8 experts):

| layer | sparse spread (max-min) | dense spread | sparse min | dense min |
|---|---|---|---|---|
| L1 | 0.074 | 0.059 | 0.082 | 0.096 |
| L2 | 0.079 | 0.062 | 0.079 | 0.087 |
| L3 | 0.082 | 0.074 | 0.095 | 0.095 |
| L4 | 0.206 | 0.204 | 0.045 | 0.042 |
| L5 | 0.118 | 0.140 | 0.083 | 0.064 |

No collapsed experts in either implementation (minimum load 0.042-0.083
against a 0.125 target; the aux loss holds). No NaN/Inf parameters in
either model. Final val loss: sparse 6.0385, dense 6.0477 (identical
within run noise; sparse marginally better).
