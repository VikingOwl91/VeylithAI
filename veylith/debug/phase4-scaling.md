# Phase 4 / Phase D: sparse vs dense scaling experiments

Environment: `phase4` venv (torch 2.16.0.dev20261006+rocm7.14), RX 9070 XT
(17.1 GB), bf16, medians over 3x50 measured iterations after 10 warmup
iters, CUDA-event timing. Raw measurements and peak reserved memory are in the [scaling results](version-isolation/scaling-results.txt).

| Configuration | Implementation | Forward (ms) | Backward (ms) | Full step (ms) | Tokens/s | Peak allocated (GB) | Run log |
|---|---|---:|---:|---:|---:|---:|---|
| b16_blk256 | sparse | 28.30 | 39.69 | 74.45 | 55,017 | 1.56 | [Completed](logs/scaling_b16_blk256_sparse.log) |
| b16_blk256 | dense | 19.97 | 33.07 | 59.33 | 69,039 | 2.38 | [Completed](logs/scaling_b16_blk256_dense.log) |
| b64_blk256 | sparse | 53.88 | 71.89 | 130.77 | 125,286 | 4.70 | [Completed](logs/scaling_b64_blk256_sparse.log) |
| b64_blk256 | dense | 49.94 | 94.02 | 148.93 | 110,010 | 8.02 | [Completed](logs/scaling_b64_blk256_dense.log) |
| b128_blk256 | sparse | 94.46 | 142.59 | 241.78 | 135,529 | 8.90 | [Completed](logs/scaling_b128_blk256_sparse.log) |
| b128_blk256 | dense | — | — | — | — | — | [Out of memory](logs/scaling_b128_blk256_dense.log) |
| b64_blk512 | sparse | 112.00 | 171.94 | 288.30 | 113,658 | 11.32 | [Completed](logs/scaling_b64_blk512_sparse.log) |
| b64_blk512 | dense | — | — | — | — | — | [Out of memory](logs/scaling_b64_blk512_dense.log) |
| b128_blk512 | sparse | — | — | — | — | — | [Out of memory](logs/scaling_b128_blk512_sparse.log) |
| b128_blk512 | dense | — | — | — | — | — | [Out of memory](logs/scaling_b128_blk512_dense.log) |
| b64_e16k4 | sparse | — | — | — | — | — | [GPU page fault (exit 134)](logs/scaling_b64_e16k4_sparse.log) |
| b64_e16k4 | dense | 70.83 | 136.86 | 215.61 | 75,988 | 13.01 | [Completed](logs/scaling_b64_e16k4_dense.log) |

At block size 256 with eight experts, dense had the shorter full step at batch 16. At batch 64, sparse reduced step time by 12.2% and peak allocated memory by 41.4%. The timing advantage came from backward (71.89 vs 94.02 ms). Sparse forward was still slower (53.88 vs 49.94 ms).

Sparse also completed two shapes where dense ran out of memory, B128 × 256 and B64 × 512. Both implementations ran out of memory at B128 × 512. With 16 experts and top-4 routing at B64 × 256, dense completed and sparse hit a GPU page fault. The cause remains open.

The grid measures GPU timing and memory. It does not establish numerical stability at the larger shapes. The [training comparison](phase4-training.md) records validation loss and end-to-end throughput for the eight-expert training runs.
