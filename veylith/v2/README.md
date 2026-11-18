# veylith v2

The Python commands in this historical guide run on `archive/python-research`. [Browse the archived directory](https://github.com/VikingOwl91/VeylithAI/tree/archive/python-research/veylith/v2)

> Research status: v2.x is an experimental MoE architecture research
> prototype, not a trained, production-ready or generally usable language
> model. Checkpoints are experiment artifacts. The v2 freeze tag is a
> research state, not a model release. See `FREEZE-2026-10-26.md`.

A miniature 2026 frontier-style transformer, still trained from scratch
on Tiny Shakespeare. Same recipe as v1, upgraded architecture:

| | v1 | v2 |
|---|---|---|
| positions | learned absolute | RoPE |
| norm | LayerNorm | RMSNorm |
| attention heads | MHA 8Q/8KV | GQA 8Q/2KV |
| context | full attention | sliding window 128, full every 4th layer |
| MLP | GELU 4x | SwiGLU 8/3x |
| FFN per layer | dense | MoE: 8 experts, top-2, +1 shared, aux load-balancing loss |
| prediction | next token | next token + token after next (MTP) |
| precision | bf16 autocast | same |

Every design choice is grounded in a config under `../reference/`
(DeepSeek-V4-Pro, GLM-5, Muse Glimmer 30B, MiMo-V2.5).

## Run

```bash
pip install -r requirements.txt          # runtime; sentencepiece only for BPE
pip install -r requirements-dev.txt      # adds pytest, on top of the above

python train.py                                           # char tokenizer
python train.py --tokenizer bpe --bpe_vocab_size 1000     # custom BPE
python sample.py                                          # generate from checkpoint
```

## Checkpoints and resume

Two files, each with one job:

| file | default | contents |
|---|---|---|
| inference artifact | `--checkpoint`, `checkpoints/ckpt.pt` | best-validation weights, config, tokenizer; what `sample.py` loads |
| training state | `--state-path`, `<checkpoint>.state.pt` | weights, optimizer state, global step, best weights, RNG state; for `--resume` |

- Writes are atomic (temp file plus `os.replace`), so a crash never leaves a
  truncated file and never damages the previous checkpoint.
- The resumable state is written every `--save-every` iterations (default:
  `--eval_interval`, `0` disables) and when `--stop-after N` pauses a run.
- `--stop-after N` pauses without writing the inference artifact; continue with
  `--resume`.
- `--resume` restores weights, optimizer, global step, best-validation state and
  RNG. On CPU the continuation matches an uninterrupted run exactly, including
  the per-iteration losses.
- Resume is refused when the training config changed. Allowed: raising
  `--max_iters` (this changes the LR schedule, so the curve is not the same),
  `--eval_interval`, `--save-every`, `--seed`, `--device` and paths. Rejected:
  every field of the model config (`n_layer`, `n_embd`, `n_experts`,
  `block_size`, `dropout`, ...), `--batch_size`, `--lr`, `--warmup_iters` and
  the tokenizer type.
- A state file is about 410 MB for the default model: weights, two Adam moments
  and the best weights.

Tests run on CPU only:

```bash
python -m pytest veylith/v2/tests veylith/tests -q
```

## Loss components and evaluation

The training objective is a composite. Model quality is the main next-token
cross-entropy. Do not read the composite as a quality number.

| component | meaning | weight |
|---|---|---|
| `main_ce` | next-token cross-entropy, position t predicts t+1 | 1.0 |
| `mtp_ce` | multi-token term, position t predicts t+2 | `--mtp_weight`, default 0.5 |
| `aux_loss` | Switch-style MoE load balancing, summed over MoE layers | `--moe_aux_weight`, default 0.01 |
| `composite_loss` | `main_ce + mtp_weight * mtp_ce + moe_aux_weight * aux_loss` | training objective |
| `main_perplexity` | `exp(main_ce)` | report only |

Every evaluation prints the composite on the historical line and the components
on a second line. A stored `val_loss` keeps its historical meaning: the
composite at the selected checkpoint. `metrics` holds the components,
`selection` the criterion and its value.

Evaluation modes (`--eval-mode`, default `legacy`):

- `legacy`: fresh random windows per evaluation, drawn from the global RNG.
  This is the historical behaviour, and it does consume the training RNG
  stream.
- `fixed`: one private generator seeded with `--eval-seed`, so every
  evaluation uses the same windows and the training RNG is untouched.
  Identical data, seed, checkpoint and config give identical CPU results, also
  across processes.

Each split is always drawn from its own token range; train and validation are
never mixed.

Checkpoint selection and early stopping (`--select-metric`, default
`composite`):

- `composite`: historical behaviour.
- `main_ce`: select on the main next-token CE.

One criterion drives both the kept best weights and early stopping. The chosen
criterion is printed at startup and stored in `selection`. Changing the eval
mode, the eval seed or the selection metric mid-run is refused on resume, since
each one changes either the data order or the selected checkpoint.

Limits: exactness is verified on CPU. On GPU the GEMM reduction order is not
deterministic, so bitwise equality is not guaranteed there.

Comparing runs: v1 and v2 are not comparable on raw loss values. v1 reports the
plain next-token CE, v2 reports a composite that includes MTP; tokenizer,
corpus and evaluation procedure differ as well. Compare `main_ce` per token only
against runs with the same tokenizer, corpus and eval mode, or compare bits per
character.

## Validation and failure modes

Errors surface before training starts and name the parameter.

- `--device`: `auto` is the historical pick (CUDA/ROCm, then MPS, then CPU).
  An explicit `cpu`, `cuda` or `mps` is never silently replaced: an unavailable
  device aborts with `--device cuda was requested but CUDA/ROCm is not available`.
- Config: `n_layer`, `n_embd`, `n_head`, `n_kv_head`, `sliding_window`,
  `full_attn_every`, `n_experts`, `n_experts_per_tok`, `first_k_dense`, `n_mtp`,
  `mlp_ratio` and `vocab_size` are checked when the model is built. `n_embd`
  must be divisible by `n_head`, `n_head` by `n_kv_head`, and the head width
  must be even for RoPE. Those three are assertions, so `python -O` disables
  them.
- MTP needs a window of at least two positions: `n_mtp >= 1` with `block_size 1`
  is rejected instead of producing a NaN loss. Without MTP, a window of one
  token remains valid.
- Training args: `--max_iters`, `--batch_size`, `--eval_iters`,
  `--eval_interval`, `--save-every`, `--warmup_iters`, `--lr`, `--stop-after`
  and `--checkpoint` are validated up front.
- Prompts: a char tokenizer has a fixed set of characters, so a prompt with any
  other character raises an explicit error naming it. The demonstration sample
  at the end of a run is skipped with that message, so a finished run still
  writes its checkpoint. BPE tokenizers use byte fallback and encode anything.

## Profiling

`--profile` measures the training loop per phase. It is off by default and
the loop without it is unchanged.

```bash
python train.py --tokenizer bpe --bpe_vocab_size 1000 \
    --profile --profile-warmup 20 --profile-iters 100 \
    --profile-json /tmp/profile.json
```

| flag | default | meaning |
|---|---|---|
| `--profile` | off | enable measurement |
| `--profile-warmup N` | 20 | iterations to skip (they still train) |
| `--profile-iters M` | 50 | iterations to record |
| `--profile-json PATH` | none | write the summary as JSON (implies `--profile`) |

Phases: `data` (batch construction plus the device copy), `forward`,
`backward` (zero_grad plus backward), `optimizer` (clip plus step). The
iteration window starts before `data` and ends after the loss read; the small
remainder that is not inside a phase is reported as `other`.

On CUDA each phase is followed by `torch.cuda.synchronize()`, so a phase's
wall-clock includes the kernels it queued. The time spent waiting inside
synchronize is reported per phase as `sync_wait_mean_ms` and is already part
of the phase time. The first measured iteration starts with a device drain so
warmup work does not leak into it. Draining at every phase boundary removes
cross-phase pipelining, so these numbers are not directly comparable with an
unsynchronized loop or with event-timed microbenchmarks.

Output: mean, median and p95 per phase plus iterations/s. The JSON contains
`iterations` (seen, warmup, measured), `steps`, `device`, `device_sync`,
`phases` and `env` (`PYTORCH_HIP_ALLOC_CONF`, torch version), so a GPU
measurement records whether the allocator setting was active.

The instrumentation itself costs about 4.4 microseconds per iteration on CPU
(measured with a tight loop). End-to-end differences on CPU runs were within
run-to-run noise.

## Notes

- MoE training uses the classic Switch-style auxiliary load-balancing
  loss. The frontier models in `../reference/` use aux-loss-free
  balancing instead (`topk_method: "noaux_tc"` in their configs).
- The MoE is dense-combined: routing is top-2, but every expert computes
  on every token and the router weights combine the results. True
  sparse routing needs variable-size indexing (gather/scatter with
  per-batch sizes), which faults intermittently on affected ROCm/PyTorch
  builds. Dense-combined costs E/k times the expert FLOPs and avoids that
  entirely. Frontier labs use custom grouped-GEMM kernels for sparse routing.
- MTP here is a training-time loss (position t also predicts t+2).
  The inference-time payoff, speculative decoding, is a serving trick
  and is not wired into `sample.py`.
- The first `--first_k_dense` layers are dense, the rest MoE, like
  GLM-5's `first_k_dense_replace`.
- Early stopping: the saved checkpoint holds the best-validation
  weights, not the final ones.
- The reported `val_loss` is the composite training objective, not a quality
  metric; see "Loss components and evaluation".
- Checkpoints written before P0-2 still load: `metrics`, `selection` and
  `evaluation` are additive fields. Training states from P0-1 resume with the
  historical defaults (`legacy`, `composite`).
