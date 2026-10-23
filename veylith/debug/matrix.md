# Reproducibility matrix

All runs on the RX 9070 XT (ROCm 7.2.4, torch 2.14.1, bf16 autocast unless
noted). "Frozen" mode = `reproducer.py --no_optimizer` (model stays at init;
makes the fault deterministic). Baseline fault point: iter ~26-49 (frozen) or
iter ~0-24 (full loop). The original `train.py` probe (real data, optimizer)
faults intermittently at ~17% per 250 iters.

## Environment / controls

| # | config | result | runs | notes |
|---|---|---|---|---|
| 1 | baseline sparse, frozen | FAULT | 8/8 | deterministic, iter ~26-49 |
| 2 | baseline sparse, full loop (optimizer) | FAULT | 5/5 | iter ~0-24 |
| 3 | original train.py probe (char, real data) | FAULT | 1/6 | intermittent (~17%/250it) |
| 4 | `--dtype fp32` | COMPLETED | 1/1 | fault is bf16-specific |
| 5 | `--first_k_dense 6` (no MoE) | COMPLETED | 1/1 | MoE necessary |
| 6 | `--n_mtp 0` | FAULT | 1/1 | MTP not necessary |
| 7 | `--n_layer 1` (dense only) | COMPLETED | 1/1 | |
| 8 | `--n_layer 2` (1 MoE layer) | COMPLETED | 1/1 | see confound note |
| 9 | `--n_experts 4` | COMPLETED | 1/1 | see confound note |
| 10 | `--n_experts_per_tok 8` (uniform routing) | COMPLETED | 1/1 | variable-size routing necessary |
| 11 | `--batch_size 16` | FAULT | 1/1 | |
| 12 | `--block_size 64 --sliding_window 32` | COMPLETED | 1/1 | see confound note |
| 13 | `--n_kv_head 8` (MHA) | COMPLETED | 1/1 | see confound note |
| 14 | `--sliding_window 256 --full_attn_every 1` (no SWA) | COMPLETED | 1/1 | see confound note |
| 15 | `--impl dense` (current v2 dense MoE) | COMPLETED | 1/1 | fault is sparse-MoE-specific |
| 16 | `--device cpu` (small config) | COMPLETED | 1/1 | GPU-specific; logic correct on CPU |
| 17 | `--no_backward` (forward only) | COMPLETED | 3/3 | fault is in the backward |
| 18 | `PYTORCH_CUDA_ALLOC_CONF=backend:native` | FAULT | 3/3 | not the caching allocator |
| 19 | `CUDA_LAUNCH_BLOCKING=1` | FAULT | 1/1 | not an async race |

## Backward bisect (`--grad_filter`, frozen mode)

Clean dimension: `grad_filter` does not change model init, data, or batches,
so all rows share the identical forward pass; only the backward differs.

| # | trainable group | result | notes |
|---|---|---|---|
| 20 | all (control) | FAULT | |
| 21 | experts only | FAULT | the fault manifests with expert-grad backward |
| 22 | gate only | COMPLETED | |
| 23 | shared only | COMPLETED | |
| 24 | attn only | COMPLETED | |
| 25 | emb+head only | COMPLETED | |
| 26 | norms only | COMPLETED | |
| 27 | experts+gate+shared | COMPLETED | adding kernels shifts layout, fault does not manifest |
| 28 | attn+emb_head+ln (no MoE params) | COMPLETED | |

## Isolated op tests (all clean)

- Per-op repro at real shapes (N=16384 tokens, E=8, k=2, C=256, H=682),
  forward+backward, 300 iters each, fp32 and bf16: topk, nonzero+index_select,
  index_add, scatter_add_, one_hot, bool index_put, bool advanced indexing,
  expert SwiGLU, and the full sparse-MoE layer forward+backward (fixed input).
- GEMM shape sweep across the whole forward path, 24 subset sizes for expert
  GEMMs, fp32 and bf16, forward+backward: all clean.

## Confound note

Rows 8-14 change the model, which shifts the RNG stream (model init draws
differ), so their single-run COMPLETED results are weak evidence on their
own. The grad_filter rows (20-28) are the clean dimension. The consistent
picture: the fault needs the sparse-MoE backward in bf16, and its
manifestation is layout-sensitive (row 27).

## Op-level traces (CUDA_LAUNCH_BLOCKING=1)

- Forward trace: forward completes; fault fires right after the head module
  hook (loss-term region) or at the next iter's H2D copy, depending on run.
- Backward trace (all grads and experts-only): the backward of the traced
  iter completes fully (last hook: block 1 experts / tok_emb); the fault
  fires during the NEXT iter's `get_batch` H2D copy.
- Fault addresses are consistently 2MB-aligned (segment bases), consistent
  with a write past the last allocation in a mapped segment into an
  unmapped one.

## Layer bisect (multi-seed, experts-only mode)

Defeats the RNG-shift confound by running 3 seeds per config.

| n_layer | MoE layers | faulted | completed | notes |
|---|---|---|---|---|
| 2 | 1 | 1/3 (seed 7, iter ~0-24) | 2/3 | not layer-count-dependent |
| 3 | 2 | 0/3 | 3/3 | |
| 4 | 3 | 0/3 | 3/3 | |
| 5 | 4 | 0/3 | 3/3 | |
| 6 | 5 | 1/3 (seed 1337, iter 25) | 2/3 | |

## dtype, multi-seed (frozen mode, 100 iters)

| dtype | seeds faulted | seeds completed | conclusion |
|---|---|---|---|
| bf16 | baseline faults (rows 1-2) | - | |
| fp32 | 0/5 | 5/5 (seeds 1337, 42, 7, 99, 123) | fault is bf16-specific, confirmed across seeds |

## Micro-reproducer (index-op level)

| test | trials | result | notes |
|---|---|---|---|
| index_select backward only, simple layout | 5000 | clean | |
| bf16 index_add backward only, simple layout | 5000 | clean | |
| combined micro-test (both backward paths, combination layout) | faults at trial 0, at the first `index_add` | FAULT | **false positive**: the micro-test used the (M, C) expert input as the `index_add` base while `sel` holds indices up to N-1, an OOB index in the micro-test itself |
| corrected micro-test (full-size base, in-bounds) | 3000 | clean | confirms the above was a micro-test bug, not the real fault |

Lesson from the false positive: an OOB index in `index_add` produces
exactly this fault signature (HSAIL hardware exception at unmapped VA).
This validates the OOB-write theory for the real fault, and confirms the
real code (in-bounds indices, audited) must be faulting inside a kernel.
