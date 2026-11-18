# Phase 4: sparse vs dense scaling

Status: measurements complete, interpretation partly open. This record fills
the PENDING placeholder of the earlier working copy.

## Method

- Environment: isolated `phase4` venv, torch 2.16.0.dev20261006+rocm7.14,
  RX 9070 XT (17.1 GB), bf16.
- Model: 25.60M parameters, 6 layers, expert width 682, 8 experts with top-2
  routing. The dense implementation is dense-combined (every expert on every
  token); batch 64 / block 256 unless stated otherwise.
- Per cell: 10 warmup iterations, then 3 blocks x 50 measured iterations,
  CUDA-event timing, medians. One process per cell.
- `sparse` is the recovered sparse routing (`sparse_moe_v2_train.py`),
  `dense` is the shipped dense-combined implementation (`v2/train.py`).
- These are benchmark steps on synthetic tokens, not training runs.

## Measured results

Median timings and peak memory from the [recorded grid](version-isolation/scaling-results.txt). Each configuration links to its run log.

| config | impl | fwd ms | bwd ms | step ms | tok/s | alloc GB | reserved GB |
|---|---|---|---|---|---|---|---|
| [b16_blk256](logs/scaling_b16_blk256_sparse.log) | sparse | 28.30 | 39.69 | 74.45 | 55,017 | 1.56 | 1.72 |
| [b16_blk256](logs/scaling_b16_blk256_dense.log) | dense | 19.97 | 33.07 | 59.33 | 69,039 | 2.38 | 2.49 |
| [b64_blk256](logs/scaling_b64_blk256_sparse.log) | sparse | 53.88 | 71.89 | 130.77 | 125,286 | 4.70 | 5.13 |
| [b64_blk256](logs/scaling_b64_blk256_dense.log) | dense | 49.94 | 94.02 | 148.93 | 110,010 | 8.02 | 8.53 |
| [b128_blk256](logs/scaling_b128_blk256_sparse.log) | sparse | 94.46 | 142.59 | 241.78 | 135,529 | 8.90 | 9.75 |
| [b128_blk256](logs/scaling_b128_blk256_dense.log) | dense | OOM | | | | | |
| [b64_blk512](logs/scaling_b64_blk512_sparse.log) | sparse | 112.00 | 171.94 | 288.30 | 113,658 | 11.32 | 13.19 |
| [b64_blk512](logs/scaling_b64_blk512_dense.log) | dense | OOM | | | | | |
| [b128_blk512](logs/scaling_b128_blk512_sparse.log) | sparse | OOM | | | | | |
| [b128_blk512](logs/scaling_b128_blk512_dense.log) | dense | OOM | | | | | |
| [b64_e16k4](logs/scaling_b64_e16k4_sparse.log) | sparse | FAILED(exit=134) | | | | | |
| [b64_e16k4](logs/scaling_b64_e16k4_dense.log) | dense | 70.83 | 136.86 | 215.61 | 75,988 | 13.01 | 13.59 |

`OOM` means `torch.cuda.OutOfMemoryError` aborted the run. `FAILED(exit=134)`
is the GPU fault class: `Memory access fault ... Page not present` (see
`README.md`).

## What the data shows

- **Crossover between batch 16 and 64** at block 256: dense is faster at
  batch 16 (59.33 vs 74.45 ms per step), sparse is faster at batch 64
  (130.77 vs 148.93 ms, 12.2% lower).
- **Memory**: at batch 64, sparse peaks at 4.70 GB allocated against 8.02 GB
  for dense (41.4% less). At batch 16 it is 1.56 vs 2.38 GB.
- **Sparse completed two configurations where dense ran out of memory**:
  batch 128 / block 256 (8.90 GB) and batch 64 / block 512 (11.32 GB).
  Both implementations ran out of memory at batch 128 / block 512.
- **16 experts / top-4 at batch 64**: dense completed (215.61 ms, 13.01 GB),
  sparse aborted with the GPU fault. That is evidence against adopting
  sparse routing, not a benchmark result in its favour.

## What is not established

- One run per cell. Medians suppress noise inside a run, not between runs;
  there was no repetition on another day or after a reboot.
- No training runs at the larger batch/block sizes, so nothing here says
  whether training stays stable at those sizes.
- Numerical stability at larger scales was not measured.
- The 16/top-4 fault from this table was not analysed as part of the
  scaling run. The separate E16/K4 investigation traced it (kernel name,
  candidate call, allocator experiment) without hardware-level confirmation
  of the mechanism; see `README.md`.
- The 12% figure here (130.77 vs 148.93 ms) comes from this table. The draft
  issue report quotes a different run (124.10 vs 140.60 ms) with the same
  percentage; the absolute step times differ by about 5% between runs.
- Wall-clock training throughput is lower than these step times imply
  (about 5.5 it/s at batch 64). CPU-side batch construction was measured at
  0.79 ms per batch, so data loading alone does not explain the gap. The
  remainder is open.

## Provenance

The [scaling driver](version-isolation/run-scaling.sh) produced the [recorded grid](version-isolation/scaling-results.txt). The [training comparison](phase4-training.md) covers validation loss and end-to-end throughput.
