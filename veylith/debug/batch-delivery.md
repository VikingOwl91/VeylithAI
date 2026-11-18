# Eight-expert training throughput with pre-staged batches

Keep the existing V2 batch loader for this workload. Preparing all inputs on the GPU before the timed window increased mean throughput by 0.66% for sparse MoE and 0.32% for dense-combined MoE in two passes. Even these small gains exclude preparation time. Both implementations used an additional 9.75 MiB of peak allocated GPU memory.

## Measurements

Each row covers 491,520 training tokens after 10 warmup steps. Rows appear in execution order. Each process started from the same seed. The second pass reversed the first pass's order.

| Pass | Model | Delivery | Tokens/s | Window (s) | Delivery (ms/step) | Staging (ms) | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Sparse | Normal | 123,261 | 3.988 | 30.848 | 0 | 4,788.46 | 5,152 |
| 1 | Dense | Pre-staged | 109,314 | 4.496 | 0.00067 | 46.48 | 8,092.39 | 8,542 |
| 1 | Sparse | Pre-staged | 123,655 | 3.975 | 0.00057 | 45.54 | 4,798.21 | 5,160 |
| 1 | Dense | Normal | 108,943 | 4.512 | 91.999 | 0 | 8,082.64 | 8,532 |
| 2 | Dense | Normal | 108,849 | 4.516 | 92.077 | 0 | 8,082.64 | 8,532 |
| 2 | Sparse | Pre-staged | 123,387 | 3.984 | 0.00058 | 47.05 | 4,798.21 | 5,160 |
| 2 | Dense | Pre-staged | 109,184 | 4.502 | 0.00071 | 45.05 | 8,092.39 | 8,542 |
| 2 | Sparse | Normal | 122,165 | 4.023 | 28.475 | 0 | 4,788.46 | 5,152 |

Delivery measures host wall time inside the normal loader or pre-staged list lookup. The normal loader's synchronous copies can wait for work submitted in the preceding step. Counting this time as removable CPU work would overstate the gain. Synchronizing at each end of the training window captures total elapsed time. Preparing all 40 batches up front took 45–47 ms, including the warmup batches.

The earlier training logs report cumulative iterations/s across training, periodic evaluation and best-state copies to CPU. This experiment measures training steps alone. Its results give little reason to rewrite the loader for B64 × 256 on this machine. Attributing the rest of the historical wall-clock gap would require measuring evaluation and checkpoint bookkeeping in the full loop. The separate V3 loader benchmark on branch `v3/data-pipeline` measures CPU corpus iteration and resume behavior on a different data path. [Phase 4 training](phase4-training.md)

## Controls and reproduction

Tested on the RX 9070 XT with PyTorch `2.16.0.dev20261006+rocm7.14`, HIP `7.14.60850` and `TORCH_CUBLAS_WORKSPACE_CACHE=0`. The reference training code at `5fbb2bf` supplies both GPT implementations, their production batch loaders and the learning-rate schedule. All arms use eight experts, top-2 routing, bf16 autocast, dropout 0.1, AdamW, gradient clipping at 1.0 and the first 40 steps of the original 1,500-step schedule. The measurement covers steps 10–39.

Each process loads the first 90% of the saved BPE-1000 Tiny Shakespeare token tensor. Model initialization uses seed 1337. The loader uses CPU seed 1338 and the same 40 batches in every arm. Each run verifies the terminal CPU RNG state against the expected draws. A separate CPU check compared actual inputs and shifted targets to the reference windows for four batches in both loaders. Both passes and delivery modes produced terminal loss 9.115201 for sparse and 9.115708 for dense.

| Identity | SHA-256 |
|---|---|
| Saved token file | `7355f3bbbac30d43e768eacae6a734d19cbdd624cc5a2e89f520450043780b6a` |
| Ordered training-window offsets | `c9541aa13cc8ac742a01f1b8ef92aed610570b3c16e7350a996e92874a22fed7` |

From the Veylith directory, use the Phase 4 Python environment and set `IDS` to the saved `debug/phase4/sparse-run/data/tinyshakespeare_bpe1000.model.ids.pt`. The benchmark prints one JSON record per process.

```bash
set -e
export TORCH_CUBLAS_WORKSPACE_CACHE=0
for pair in sparse:normal dense:prestaged sparse:prestaged dense:normal \
            dense:normal sparse:prestaged dense:prestaged sparse:normal; do
    python debug/batch_delivery.py --ids "$IDS" \
        --impl "${pair%:*}" --delivery "${pair#*:}"
done
```
