# Training audit: why does the 25.6M MoE trail the 5.3M dense model?

Date: 2026-10-26. CPU-only, no training, no GPU. Numbers come from
`veylith/evaluation/audit.json`, produced by `veylith/audit.py`, plus the
checkpoint files, the commit messages and the documentation. Evidence is
labelled: measured here, stored in a checkpoint, reconstructed from code and
args, reported by an earlier log, or not determinable.

## 1. What the checkpoints actually record

v1 and v2 checkpoints use the legacy format: weights, config, tokenizer,
training args and a stored validation loss. They carry no step, no metrics and
no selection record. The P1-1 checkpoint (120 steps) is format 2 and does.

| checkpoint | family | tokenizer | total params | stored loss | note |
|---|---|---|---|---|---|
| v1-char | v1 | char 65 | 4,837,953 | 1.4749 | plain val CE |
| v1-bpe | v1 | BPE 1000 | 5,317,608 | 3.2903 | plain val CE |
| v1-bpe-5000it | v1 | BPE 1000 | 5,317,608 | 4.1756 | same config/args as v1-bpe, much worse |
| v2-bpe | v2 | BPE 1000 | 25,602,304 | 6.0358 composite | no step or metrics stored |
| v2-p1-1-temp | v2 | BPE 1000 | 25,602,304 | 7.3990 composite | step 120, format 2 |

## 2. Reconstructed training history

| property | v1-bpe | v2-bpe | evidence |
|---|---|---|---|
| steps | 5000 | 3000 | stored `args.max_iters` |
| batch x block | 64 x 256 | 64 x 256 | stored args and config |
| tokens per step | 16,384 | 16,384 | derived |
| token positions | 81,920,000 | 49,152,000 | derived (steps x 16,384) |
| training tokens | 443,556 | 443,556 | corpus split (same BPE) |
| corpus passes | 184.7 | 110.8 | derived |
| learning rate | 3e-4 | 3e-4 | stored `args.lr` |
| warmup | 100 | 100 | stored `args.warmup_iters` |
| schedule | linear warmup, cosine to 10 % | same | code (`lr_at`, max_iters from args) |
| dropout | 0.1 | 0.1 | stored config |
| optimizer | AdamW, betas 0.9/0.999, eps 1e-8, wd 0.01 | same | code, not stored |
| eval frequency | every 500 steps, 50 random windows | same | stored args |
| selection metric | plain val CE | composite loss | code, plus the commit message |
| selected step | **not determinable** | 500 (commit 347aa70) | commit message for v2 only |
| exposure at selection | not determinable | 8,192,000 tokens, 18.5 passes | derived from the commit message |
| main CE at selection | not determinable | 3.539 nats/token (2.256 bits/char) | commit message |
| MTP CE / aux at selection | none (v1 has no MTP) | 4.89 implied / aux ~5.09 from a later measurement | derived from the commit message |
| final evaluation | not determinable | val 8.06, train 1.39 at step 3000 | repository README, quoted from the run log (not stored) |
| learning curve | **not available** | **not available** (only the two reported points) | no logs in the repository or in git history |

Not determinable: v1's selected step, v1's evaluation points beyond the stored
best, and any intermediate curve for either model. No training logs for v1 or
v2 are tracked in git; the only curve-like statements are the v2 commit
message ("iter-500 weights") and the README (val 8.06 at iter 3000).

Consequence of the missing data: the audit cannot compare both models at the
same point in their curves. Any statement about "undertrained" or "overfit"
below is anchored either to the selected checkpoints or to the two reported
v2 points.

## 3. Train/validation gap (measured)

Protocol: same six fixed validation sections as P1-2 (3,067 characters, start
1,008,119), plus six fixed training sections of 512 characters each starting
at character 50,000 (all before the validation split). Main next-token CE only,
same chunking, CPU, float32. Sections are disjoint and hashed in `audit.json`.

| checkpoint | train CE | val CE | gap (nats) | train bits/char | val bits/char | gap (bits) |
|---|---|---|---|---|---|---|
| v1-char | 1.0786 | 1.3262 | +0.2476 | 1.5561 | 1.9133 | +0.357 |
| v1-bpe | 2.4964 | 2.9671 | +0.4707 | 1.6059 | 1.9219 | +0.316 |
| v1-bpe-5000it | 0.8584 | 3.7158 | +2.8575 | 0.5522 | 2.4069 | +1.855 |
| v2-bpe | 2.7552 | 3.1800 | +0.4248 | 1.7724 | 2.0598 | +0.287 |
| v2-p1-1-temp | 4.6927 | 4.6701 | -0.0226 | 3.0188 | 3.0250 | +0.006 |

Reading:

- **v2 is not more overfit than v1 at the shipped checkpoint.** The gaps are
  0.425 vs 0.471 nats (0.287 vs 0.316 bits/char). v2 fits the training text
  *worse* (2.755 vs 2.496) and the validation text worse (3.180 vs 2.967).
- **`v1-bpe-5000it` is a late, heavily memorised checkpoint** (train 0.86,
  val 3.72). It is not the v1 reference; its origin is not recorded.
- **The 120-step model is undertrained** (both sides ~4.67, no gap).

So the shipped v2 checkpoint is on the *underfit* side of its own curve: it
has not yet matched v1's training fit, and its validation loss had already
stopped improving. The two reported points from the run log say what happens
next: by step 3000 the training loss fell to 1.39 while validation rose to
8.06. The useful window is narrow and the shipped checkpoint sits at its early
edge.

## 4. Loss decomposition: value and gradient

Measured on the six validation sections with the v2 checkpoint (main CE,
MTP CE and aux from the same forward pass):

| term | raw | weight | weighted | share of composite |
|---|---|---|---|---|
| main CE | 3.1800 | 1.0 | 3.1800 | 56.0 % |
| MTP CE | 4.8904 | 0.5 | 2.4452 | 43.1 % |
| aux | 5.1894 | 0.01 | 0.0519 | 0.9 % |
| composite | | | 5.6771 | 100 % |

Documented training-time values (commit message: composite 6.0358, main CE
2.256 bits/char): main 3.5390 (58.6 %), MTP 4.8917 (40.5 %), aux 5.0889
(0.8 %). The two decompositions agree: MTP carries about 40 to 43 % of the
composite *value*.

Gradient contribution, measured with one separate backward pass per term on
two validation chunks (no optimizer step, weights verified unchanged):

| quantity | chunk 1 | chunk 2 |
|---|---|---|
| ‖g_main‖ | 6.63 | 6.95 |
| ‖g_mtp‖ | 6.75 | 6.94 |
| ‖0.5 g_mtp‖ | 3.38 | 3.47 |
| ‖0.01 g_aux‖ | 0.033 | 0.027 |
| ‖g_composite‖ | 8.13 | 8.75 |
| cos(g_main, g_mtp) | **+0.240** | **+0.339** |
| cos(g_main, g_aux) | +0.093 | +0.004 |
| cos(g_composite, g_main) | 0.927 | 0.940 |
| MTP share of composite norm | 41.5 % | 39.7 % |

Interpretation, kept strictly separate:

- **Measured:** the weighted MTP term contributes about 40 % of the composite
  gradient norm, and its direction is *partially aligned* with the main
  gradient (cosine +0.24 to +0.34), not opposing it. The composite gradient is
  dominated by the main direction (cosine 0.93).
- **Not established:** whether this helps or hurts the main objective. A large,
  partially aligned secondary objective can act as a regulariser or as a
  competitor; the value share alone proves neither. Only an ablation answers
  it.
- `mtp_weight` is a runtime config value (CLI `--mtp_weight`, read in the
  forward pass). `mtp_weight=0` makes the gradient contribution exactly zero
  but still computes the MTP loss and its graph. `--n_mtp 0` skips the term
  entirely and is the clean "off" switch. No ablation was run.

## 5. Parameters and active capacity

Convention: dense matrices and the full embedding tables count as active (not
only the touched row); exactly `top_k` routed experts are active per token;
the router, shared expert, norms, attention and the dense FFN are always
active. MTP reuses the trunk and the output head, so it adds **zero**
parameters; the active count is identical for main-only inference and for
MTP training.

| quantity | v1-bpe | v2-bpe |
|---|---|---|
| total trainable | 5,317,608 | 25,602,304 |
| always active | 5,317,608 | 4,651,264 |
| routed experts, total | 0 | 20,951,040 |
| parameters per routed expert | - | 523,776 |
| routed experts active per token | - | 2 of 8 per layer, 5 layers |
| active per token (main) | 5,317,608 | **9,889,024** |
| active share of total | 100 % | 38.6 % |
| total / active | 1.00 | 2.59 |
| ratio to v1, total | 1.00 | 4.81 |
| ratio to v1, active | 1.00 | 1.86 |

Category breakdown (trainable parameters):

| category | v1-bpe | v2-bpe |
|---|---|---|
| token embedding | 256,000 | 256,000 |
| position embedding | 65,536 | 0 (RoPE, non-persistent) |
| attention | 1,579,008 | 983,040 |
| norms | 6,656 | 3,328 |
| dense FFN | 3,153,408 (6 layers) | 523,776 (layer 0 only) |
| shared expert | 0 | 2,618,880 |
| routed experts | 0 | 20,951,040 |
| router | 0 | 10,240 |
| output head | 257,000 | 256,000 |

v2 has *fewer* always-on parameters than v1 (4.65M vs 5.32M) and its extra
capacity comes from two routed experts per layer. Per token it activates 1.86
times v1's parameters; in total it stores 4.81 times as many. The MoE adds
total capacity, not proportional per-token compute.

## 6. Data budget and repetition

| quantity | v1-bpe | v2-bpe | v2 selected checkpoint |
|---|---|---|---|
| steps | 5000 | 3000 | 500 |
| windows drawn | 320,000 | 192,000 | 32,000 |
| possible window starts | 443,301 | 443,301 | 443,301 |
| mean draws per start | 0.72 | 0.43 | 0.07 |
| expected unseen starts | 48.6 % | 64.9 % | 93.2 % |
| token positions | 81.92M | 49.15M | 8.19M |
| corpus passes (token level) | 184.7 | 110.8 | 18.5 |

Sampling is uniform random windows, not epochs. Two different repetition
measures matter and are often conflated: at the *window-start* level most
starts are never drawn (93 % for the selected v2 checkpoint), while at the
*token* level every token is covered about 18.5 times because windows overlap.
The draw counts are expectations under uniform sampling; the actual RNG
sequence of the original runs was not recorded, so exact per-window counts are
not determinable.

## 7. Attention layout

| property | v1 | v2 |
|---|---|---|
| full-attention layers | all 6 | layer 3 only |
| sliding-window layers | none | 0, 1, 2, 4, 5 |
| window | - | 128 positions (including the current one) |
| MTP context | - | same trunk and same head, no separate context |

The implementation computes the full `T x T` attention matrix in every layer
and only masks it, so the sliding window does **not** save compute; it only
restricts which positions a layer can use. The theoretical receptive field is
still the full prefix, because the single full layer at depth 3 sees
everything and its outputs are carried by the two later SWA layers. Whether
the model actually uses long-range information is not measured. A fair
comparison is a config-only change (`--full_attn_every 1` makes every layer
full); no code change, same FLOPs, slightly larger mask work, same memory.

## 8. What the evidence says about the six candidate explanations

| explanation | evidence | verdict |
|---|---|---|
| v2 undertrained | selected at 18.5 corpus passes; train CE 2.755 vs v1 2.496; later train loss 1.39 | true at the selected point, but more training does not help validation (8.06 at step 3000) |
| v2 overfit | train/val gap 0.425, same as v1's 0.471; later 1.39 vs 8.06 | not at the shipped checkpoint; strongly later in the run |
| composite objective hurts main CE | MTP is 43 % of the value and 40 % of the composite gradient norm, but the direction is positively aligned (+0.24..+0.34) | plausible, not demonstrated; needs the ablation |
| too much capacity for the data | 25.6M total vs 5.3M, 9.9M active, rapid overfitting after step 500 | plausible and consistent, but the selected checkpoint is not more overfit than v1 |
| attention limits language modelling | 5 of 6 layers see 128 tokens; v1 sees everything | plausible and untested; config-only experiment |
| different training budgets explain the gap | v1 up to 5000 steps / 81.9M tokens; v2 3000 / 49.2M, selected at 8.2M; v1's selected step unknown | a real confound; cannot be separated from the checkpoint data alone |

## 9. Ablation plan (three experiments, in priority order)

All three share the same protocol: dense-combined v2 configuration, BPE 1000,
batch 64, block 256, seed 1337, AdamW as in the code, fixed evaluation,
selection on main CE, checkpoints and logs under `/tmp/opencode/p13/`, the
allocator env `PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:4` from the GPU
investigation as an optional workaround candidate (it is not a stability
guarantee), abort on any fault, OOM or hang, no automatic retries.

### E1: clean v2 baseline at the full budget (recommended first)

- **Hypothesis:** part of the 0.213 nats/token gap is a selection and
  evaluation artefact. The shipped v2 was selected by the composite loss on
  random evaluation windows and stopped at step 500. Under a fixed evaluation
  and main-CE selection at v1's maximum budget (5000 steps), v2's best main CE
  is lower than 3.18 and may approach v1's 2.967.
- **Controls:** data, tokenizer, batch, block, seed, LR schedule, model config,
  dtype.
- **Variable:** evaluation mode (fixed), selection metric (main CE), training
  budget (3000 to 5000 steps).
- **Primary metric:** main CE and bits/char on the P1-2 test sections at the
  selected checkpoint.
- **Secondary:** main/MTP/aux curves every 250 steps, best step, train/val gap
  at the best step, throughput, router shares at the best step.
- **Code changes:** one small addition, `--eval-skip-tokens N` (or an
  equivalent section-based selection set), so that the selection windows are
  disjoint from the six P1-2 test sections. Without it, roughly 3 % of the
  selection windows overlap the test sections. Small and CPU-testable; no
  architecture or loss change.
- **Budget:** 5000 steps x 16,384 tokens = 81.92M token positions, about
  14 minutes at the measured 6.2 iterations/s plus evaluation and one
  checkpoint.
- **GPU risks:** dense-combined has been stable; the allocator ENV is a
  candidate workaround only. Abort on fault/OOM/hang.
- **What a positive/negative result means:** best main CE at or below 2.967
  (v1's measured value on the same sections) means the earlier gap was
  selection/protocol and v2 is competitive at equal budget. A best clearly
  above 2.967 means the deficit is real at a matched budget, and the
  architecture/objective experiments E2/E3 follow. Either outcome produces the
  missing v2 curve.

### E2: capacity matched to the data (n_embd 192)

- **Hypothesis:** 25.6M total parameters on 0.44M training tokens overfit as
  soon as the model fits (train 1.39 vs val 8.06); a smaller v2 with about
  v1's active size reaches a lower validation minimum.
- **Controls:** everything as in E1 (same budget, protocol, data, seed);
  only `--n_embd 192` changes, which sets head_dim 24, expert width 512 and
  total parameters to 14.5M (active 5.67M, close to v1's 5.32M).
- **Variable:** model capacity.
- **Primary metric:** best main CE / bits/char on the P1-2 sections.
- **Secondary:** curve shape and overfitting onset, throughput, router shares.
- **Code changes:** none, config only.
- **Budget:** 5000 steps, about 12 to 13 minutes (smaller model).
- **GPU risks:** same as E1; the config passes the existing validation.
- **What it means:** if the smaller v2 beats the full v2 and approaches v1,
  capacity/data is confirmed as the main lever. If it does not, capacity is not
  the binding constraint and the attention/objective experiments dominate.

### E3: sliding window versus full attention

- **Hypothesis:** restricting five of six layers to 128 positions costs
  language-modelling quality at a 256-token context; full attention improves
  both train fit and validation.
- **Controls:** everything as in E1; only `--full_attn_every 1` changes (all
  six layers full).
- **Variable:** attention pattern.
- **Primary metric:** best main CE / bits/char on the P1-2 sections.
- **Secondary:** step time and memory (the implementation computes T x T
  attention either way, so the cost should be nearly unchanged), curves,
  router shares.
- **Code changes:** none, config only.
- **Budget:** 5000 steps, about 14 minutes.
- **GPU risks:** same as E1; attention memory at batch 64, block 256 is small.
- **What it means:** if full attention closes a substantial part of the gap,
  the window was limiting; if it does not, the window is not the cause at this
  budget.

Not in the top three: longer training (the reported val already rises), a
dense-FFN control (needs a code change and tests the parameterisation rather
than the diagnosis), and more data (no download without a separate decision).
They are the natural follow-ups if E1 to E3 do not explain the gap.

## 10. The eight required answers

1. **Is v2 demonstrably undertrained?** At the shipped checkpoint, yes in the
   fit sense: 18.5 corpus passes, train CE 2.755 against v1's 2.496, and the
   run log reports train 1.39 later. But not in the sense that more training
   would improve validation: the reported validation loss rises to 8.06 by
   step 3000. So the checkpoint is early, and more training at this config is
   not the fix.
2. **Is v2 demonstrably overfit?** Not at the shipped checkpoint: its
   train/val gap (0.425) is slightly smaller than v1-bpe's (0.471). It is
   strongly overfit later in the run (reported 1.39 vs 8.06). The shipped
   checkpoint is the underfit edge of a narrow useful window.
3. **How large is the main-CE train/val gap?** v2: 0.4248 nats/token (0.287
   bits/char); v1-bpe: 0.4707 (0.316); v1-char: 0.2476 (0.357);
   v1-bpe-5000it: 2.8575 (1.855); v2 temp: -0.0226.
4. **How many parameters are active per token in the main forward pass?**
   9,889,024 for v2 (top-2 of 8 experts in 5 MoE layers; 4,651,264 always on),
   against 25,602,304 total (ratio 2.59). v1-bpe: 5,317,608 active, all of its
   parameters. Active v2/v1 = 1.86; total v2/v1 = 4.81. MTP adds no
   parameters, so the count is identical for main-only and MTP training.
5. **What role does MTP play in the actual optimisation target?** Numerically
   43 % of the composite value (40.5 % in the documented training-time
   values); its weighted gradient contributes about 40 % of the composite
   gradient norm, and its direction is positively aligned with the main
   gradient (+0.24 to +0.34), with the composite 93 % aligned with main. A
   large but partially aligned secondary objective. Whether it helps or hurts
   is not established. `mtp_weight=0` zeroes the gradient but still computes
   the term; `--n_mtp 0` removes it completely.
6. **How much training did v1 and v2 receive?** Maximum budgets: v1 5000
   steps, 81.92M token positions, 184.7 corpus passes; v2 3000 steps, 49.15M
   positions, 110.8 passes. v2's shipped checkpoint was selected at step 500:
   8.19M positions, 18.5 passes. v1's selected step is not determinable.
7. **Which differences complicate a fair comparison?** v1's selection step is
   unknown; v1 selected on plain val CE and v2 on the composite, both on random
   windows; v1 has no MTP, no MoE, no SWA, no GQA and uses learned position
   embeddings; active parameters differ by 1.86x and total by 4.81x; the
   maximum budgets differ by 1.67x in tokens (and 10x between v2's selected
   exposure and v1's maximum); neither model has a stored learning curve.
8. **Which single experiment gives the most insight?** E1: a clean v2 run at
   v1's maximum budget with fixed evaluation and main-CE selection, scored on
   the same six test sections. It removes the selection and evaluation
   confound, produces the missing curve and decides whether the deficit is
   real at a matched budget. Only then do the architecture experiments E2/E3
   carry interpretable evidence.

## 11. Reproducing

```bash
python3 veylith/audit.py --out veylith/evaluation/audit.json
python3 -m pytest veylith/v2/tests veylith/tests -q
```

The audit is byte-identical across runs (checked twice); checkpoint and
section hashes are in `audit.json`. The gradient analysis reads weights only
and verifies that no parameter changed.
