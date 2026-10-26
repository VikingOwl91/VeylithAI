# Controlled v2 run: 5000 steps, main-CE selection, unchanged test

Date: 2026-10-26. One approved GPU training run, CPU-only preparation and a
CPU final evaluation. Raw numbers: `veylith/evaluation/p1-3-results.json`,
`veylith/evaluation/p1-3-audit.json` and
`veylith/evaluation/p1-3-curve.jsonl`. The checkpoint stays outside the
repository at `~/veylith-runs/p1-3-20261026/ckpt_best.pt`
(102,484,435 bytes, sha256 `be2d72912b222e8e2ca0df4bdca802b24f01bdb3cc694aaec789156b31e77f3f`).

## Verdict

The run is technically clean: exactly one GPU run, exit 0, no fault, no OOM,
no NaN/Inf, 5000 steps, complete curve, working checkpointing and a
byte-reproducible CPU evaluation.

The stretch goal, reaching v1-BPE, is not reached. On the unchanged six test
sections the new checkpoint scores 3.1581 main CE / 2.0456 bits/char:

| comparison | nats/token | bits/char |
|---|---:|---:|
| new v2 minus shipped v2 | -0.0237 +/- 0.0264 | -0.0142 +/- 0.0167 |
| new v2 minus v1-BPE | **+0.1890 +/- 0.0336** | **+0.1237 +/- 0.0227** |

The improvement over the shipped v2 is inside one standard error (95 % CI
-0.0915..+0.0441) and cannot be separated from run-to-run noise. The deficit
against v1-BPE is 5.6 standard errors and remains a real gap. The run also
shows the mechanism: validation CE on the fixed selection sections bottoms at
step 500 (3.2779) and rises to 4.7500 at step 4999 while training CE falls
from 2.660 to 0.588. More steps do not help this model on this corpus;
selection has to stop at the early minimum.

## Provenance

| item | value |
|---|---|
| branch | `v2/dev-2026-10-26` |
| code commit | `2108b4142eab6e5a6cae37d86092c4b2db4172bd` |
| git status at start | clean (0 entries) |
| torch | 2.14.1, HIP 7.2.53211 |
| python | 3.14.8 |
| kernel | Linux 7.2.11-zen1-1.1-zen |
| GPU | AMD Radeon RX 9070 XT, 16 GB |
| ENV | `PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:4` |
| precision | bfloat16 autocast |
| rocm-smi before start | no KFD process, 1.89 GB VRAM in use |

The allocator ENV is still a workaround candidate, not a proven fix.

## CPU preparation (Phase B)

The following mechanisms were added before the run and committed as
`2108b41`:

- `--eval-mode sections --selection-sections FILE` evaluates fixed sections
  instead of random windows. `veylith/evaluation/selection-sections.json` pins
  the corpus hash and every section hash (train 50,000..53,712, val
  1,050,000..1,053,712, width 512, gap 128, six sections each). Loading
  verifies both hashes and the BPE roundtrip; a resume with a different file
  is rejected.
- The chosen step is stored as `selection.step` in the artifact and in the
  resume state. `--curve-jsonl` appends one line per evaluation with lr,
  throughput, train/val main, MTP, aux and composite, the selection value and
  the current best step.
- `evaluate.py` and `audit.py` gained `--extra-checkpoint NAME=PATH`, so the
  final numbers come from the unchanged P1-2 protocol and the unchanged audit.

Disjointness, checked by a test:

| set | character span | used for |
|---|---|---|
| train sections | 50,000..53,712 | curve only |
| P1-2 test sections | 1,008,119..1,011,831 | final evaluation only |
| selection sections | 1,050,000..1,053,712 | checkpoint selection only |

Training uses the train split (before both validation starts). The selection
sections sit in the validation region after the test sections. The six test
sections were not read before training finished and the checkpoint was
selected.

Validation: `python3 -m pytest veylith/v2/tests veylith/tests -q` gives 89
passed. A CPU smoke with the exact real configuration (2 steps, `--device
cpu`) produced the expected curve and a checkpoint whose config is identical
to the shipped v2 checkpoint, so no architecture value drifted.

## The GPU run (Phase C)

Exact command, started once with the allocator ENV already in the process
environment:

```bash
cd ~/git/AI-Research-v2dev/veylith/v2
PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:4 python3 train.py \
  --device cuda --tokenizer bpe --bpe_vocab_size 1000 \
  --n_layer 6 --n_head 8 --n_kv_head 2 --n_embd 256 --block_size 256 \
  --sliding_window 128 --full_attn_every 4 --dropout 0.1 \
  --n_experts 8 --n_experts_per_tok 2 --first_k_dense 1 \
  --n_mtp 1 --mtp_weight 0.5 --moe_aux_weight 0.01 \
  --batch_size 64 --max_iters 5000 --lr 3e-4 --warmup_iters 100 --seed 1337 \
  --eval-mode sections --selection-sections ../evaluation/selection-sections.json \
  --eval_interval 250 --select-metric main_ce --save-every 500 \
  --checkpoint ~/veylith-runs/p1-3-20261026/ckpt_best.pt \
  --curve-jsonl ~/veylith-runs/p1-3-20261026/curve.jsonl
```

| item | value |
|---|---|
| exit code | 0 |
| steps | 5000 (0..4999) |
| wall time | 15:24:02Z to 15:37:36Z, 814 s (in-loop 807 s) |
| throughput | 6.20 it/s end to end, 6.1..6.2 it/s per interval |
| training tokens | 81,920,000 (5000 x 64 x 256) |
| corpus passes | 166.2 over 492,841 BPE tokens |
| peak allocated | 0.7 GB |
| best selection step | 500 |
| selection value | 3.2779 main CE (ppl 26.52) on the selection sections |
| artifact | `step` 5000, `selection.step` 500, `val_loss` 5.7912 (composite at the selected weights) |
| state | `ckpt_best.state.pt`, last periodic save at step 4500, best_step 500, optimizer and RNG state present |

The 6.2 it/s agrees with the P1-1 profile (162 ms/iter, 6.17 it/s), so the
dense-combined path behaved as measured. The model line reports 25.59M
"compute per token" because the dense-combined path evaluates all experts;
the algorithmic active count is 9.89M (see the audit). No fault message, no
OOM, no non-finite metric, no automatic retry.

Checkpoint semantics: the artifact holds the weights of the selected step
(500) but records the last trained step (5000). `selection.step` is the
authoritative selection record. The config in the artifact is identical to
the shipped v2 checkpoint field by field, and the tokenizer info matches, so
the run changed no architecture value.

## Learning curve

Evaluations every 250 steps, plus the final step 4999. Steps 0, 250 and 500
were successive improvements; the marker is the final best.

| step | lr | train main | train MTP | train aux | train comp | val main | val MTP | val aux | val comp | it/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.00e-06 | 6.9569 | 6.9573 | 6.2208 | 10.4978 | 6.9695 | 6.9730 | 6.2503 | 10.5185 | 0.44 |
| 250 | 2.99e-04 | 3.4389 | 5.1081 | 5.1256 | 6.0442 | 3.6153 | 5.0306 | 5.0762 | 6.1814 | 5.91 |
| 500 | 2.96e-04 | 2.6599 | 4.6699 | 5.1968 | 5.0469 | 3.2779 (best) | 4.9239 | 5.1364 | 5.7912 | 6.03 |
| 750 | 2.88e-04 | 1.9333 | 4.1442 | 5.2387 | 4.0578 | 3.4722 | 5.0728 | 5.2315 | 6.0609 | 6.09 |
| 1000 | 2.78e-04 | 1.3487 | 3.6597 | 5.2867 | 3.2315 | 3.8224 | 5.3495 | 5.2749 | 6.5499 | 6.11 |
| 1250 | 2.65e-04 | 0.8312 | 3.1281 | 5.2804 | 2.4480 | 4.1033 | 5.6090 | 5.2547 | 6.9604 | 6.13 |
| 1500 | 2.49e-04 | 0.6953 | 2.6890 | 5.2997 | 2.0928 | 4.2662 | 5.7923 | 5.2682 | 7.2150 | 6.13 |
| 1750 | 2.31e-04 | 0.5956 | 2.3540 | 5.2890 | 1.8255 | 4.4462 | 6.0071 | 5.2536 | 7.5023 | 6.14 |
| 2000 | 2.12e-04 | 0.5789 | 2.1508 | 5.2998 | 1.7073 | 4.5344 | 6.2019 | 5.2646 | 7.6880 | 6.14 |
| 2250 | 1.91e-04 | 0.5767 | 1.9632 | 5.2856 | 1.6111 | 4.5908 | 6.2736 | 5.2718 | 7.7803 | 6.15 |
| 2500 | 1.69e-04 | 0.5849 | 1.7743 | 5.3038 | 1.5251 | 4.6625 | 6.3748 | 5.2589 | 7.9025 | 6.15 |
| 2750 | 1.48e-04 | 0.6093 | 1.6284 | 5.2919 | 1.4764 | 4.6747 | 6.4262 | 5.2753 | 7.9405 | 6.15 |
| 3000 | 1.27e-04 | 0.5851 | 1.6098 | 5.2957 | 1.4429 | 4.6732 | 6.4969 | 5.2668 | 7.9743 | 6.15 |
| 3250 | 1.06e-04 | 0.5699 | 1.5573 | 5.2895 | 1.4014 | 4.7233 | 6.5711 | 5.2705 | 8.0616 | 6.16 |
| 3500 | 8.78e-05 | 0.5865 | 1.4886 | 5.2930 | 1.3837 | 4.7488 | 6.6010 | 5.2653 | 8.1020 | 6.17 |
| 3750 | 7.11e-05 | 0.5852 | 1.4702 | 5.2876 | 1.3732 | 4.7320 | 6.6108 | 5.2755 | 8.0902 | 6.19 |
| 4000 | 5.68e-05 | 0.5784 | 1.4326 | 5.2924 | 1.3476 | 4.7180 | 6.6247 | 5.2738 | 8.0830 | 6.20 |
| 4250 | 4.53e-05 | 0.5827 | 1.4318 | 5.2909 | 1.3515 | 4.7386 | 6.6582 | 5.2743 | 8.1204 | 6.21 |
| 4500 | 3.69e-05 | 0.5841 | 1.4065 | 5.2868 | 1.3402 | 4.7345 | 6.6727 | 5.2686 | 8.1236 | 6.20 |
| 4750 | 3.17e-05 | 0.5847 | 1.3933 | 5.2890 | 1.3343 | 4.7644 | 6.6890 | 5.2706 | 8.1616 | 6.20 |
| 4999 | 3.00e-05 | 0.5878 | 1.3861 | 5.2957 | 1.3338 | 4.7500 | 6.6958 | 5.2736 | 8.1507 | 6.20 |

Reading: the selection CE falls from 6.9695 to 3.2779 at step 500, then rises
monotonically to about 4.7. Training CE falls from 6.957 to 0.588 and the MTP
training CE from 6.957 to 1.386, both pure memorization after step 500. The
aux term stays at 5.1..5.3 and does not explain the composite rise. The
validation composite bottoms at 5.7912 at step 500, so composite selection and
main-CE selection pick the same step in this run.

## Final evaluation on the unchanged test sections

Command (CPU only):

```bash
cd ~/git/AI-Research-v2dev
python3 veylith/evaluate.py --out veylith/evaluation/p1-3-results.json \
  --extra-checkpoint p1-3-v2=~/veylith-runs/p1-3-20261026/ckpt_best.pt
```

Protocol unchanged: the same six sections starting at character 1,008,119
(512 chars, 128 char gap), main next-token CE only, 3067 evaluated characters
and 1377 BPE tokens, bits/char over the exact character spans.

| model | main CE | perplexity | nats/char | bits/char |
|---|---:|---:|---:|---:|
| v1-char | 1.3262 | 3.77 | 1.3262 | 1.9133 |
| v1-BPE | 2.9671 | 19.44 | 1.3322 | 1.9219 |
| v1-BPE 5000it | 3.7158 | 41.09 | 1.6683 | 2.4069 |
| v2-BPE shipped | 3.1800 | 24.05 | 1.4277 | 2.0598 |
| v2-P1-1 120 steps | 4.6701 | 106.71 | 2.0967 | 3.0250 |
| **v2-BPE P1-3 (step 500)** | **3.1581** | **23.53** | **1.4179** | **2.0456** |

Paired over the six sections (new minus reference), 95 % CI with t(5):

| pair | nats/token | bits/char |
|---|---:|---:|
| new v2 - shipped v2 | -0.0237 +/- 0.0264, CI -0.0915..+0.0441 | -0.0142 +/- 0.0167, CI -0.0571..+0.0288 |
| new v2 - v1-BPE | +0.1890 +/- 0.0336, CI +0.1028..+0.2753 | +0.1237 +/- 0.0227, CI +0.0652..+0.1821 |

For reference, the shipped v2 was +0.2127 +/- 0.0381 nats/token behind v1-BPE
in the P1-2 report; those numbers reproduce exactly from the same payload.

## What changed, and what cannot be attributed

- **Longer training**: no improvement. The selected step is 500 of 5000. The
  extra 4500 steps only fit the training split and worsen both validation
  measures.
- **Changed selection**: no measurable effect in this run. Main CE and
  composite have their minimum at the same step (500), so `--select-metric
  main_ce` picked the same step composite selection would have picked. The
  small difference against the shipped checkpoint is a different training
  trajectory (the historical run used legacy random-window evaluation, which
  consumes the training RNG, so its checkpoints differ even at the same step),
  not a selection effect.
- **Generalization to the independent test sections**: a small improvement
  (-0.024 nats/token), not significant at six sections. The direction is
  consistent with the slightly better train fit, but the sample is too small
  to claim a gain.
- **Router behaviour**: unchanged within noise, see below. No collapse, no
  expert death.
- **Overfitting**: strong and early. The fixed-section gap (train to test) is
  +0.4977 nats for the new checkpoint vs +0.4248 for the shipped v2 and
  +0.4707 for v1-BPE. The new model fits the train sections better (2.660 vs
  2.755) and generalizes slightly better (3.158 vs 3.180); its gap is larger
  only because the train side dropped more.

Because training budget and selection changed together in the original plan,
and because in this run the selection criterion made no difference, no part of
the small test improvement can be assigned to the selection change. It is
trajectory variation around the same architecture and data.

## Train/val gap on fixed sections

From `veylith/evaluation/p1-3-audit.json` (six fixed train sections at 50,000,
six fixed test sections at 1,008,119, main CE only):

| model | train main CE | val main CE | gap |
|---|---:|---:|---:|
| v1-char | 1.0786 | 1.3262 | +0.2476 |
| v1-BPE | 2.4964 | 2.9671 | +0.4707 |
| v1-BPE 5000it | 0.8584 | 3.7158 | +2.8575 |
| v2-BPE shipped | 2.7552 | 3.1800 | +0.4248 |
| **v2-BPE P1-3** | **2.6604** | **3.1581** | **+0.4977** |

Loss decomposition at the selected checkpoint: main 55.9 %, weighted MTP
43.2 %, weighted aux 0.92 % of the composite. Gradient cosine between the main
and MTP terms is +0.220 (shipped v2: +0.240), so MTP is still aligned with the
main objective but still carries a large share of the value and gradient.

## Router

All eight experts are used in all five MoE layers; no unused expert. Values on
the six test sections:

| layer | entropy (normalised) | min share | max share |
|---|---:|---:|---:|
| 1 | 0.810 | 9.6 % | 15.9 % |
| 2 | 0.813 | 10.2 % | 15.8 % |
| 3 | 0.833 | 10.4 % | 16.5 % |
| 4 | 0.798 | 9.8 % | 15.8 % |
| 5 | 0.776 | 8.1 % | 14.9 % |

Shipped v2 for comparison: 0.763..0.841 entropy, 9.2..16.6 % shares. The
router did not change materially; the P1-2 finding that selection depends on
the text section still applies. This run gives no evidence that training
length fixes or worsens routing.

## Generation

Unchanged settings (greedy, 100 tokens, same prompts and seeds; 24 runs per
model). Aggregate repetition, mean over runs:

| model | rep3 | rep4 | unique token ratio | flagged runs |
|---|---:|---:|---:|---:|
| shipped v2 | 0.099 | 0.075 | 0.598 | 1 |
| **P1-3 v2** | **0.100** | **0.074** | **0.600** | **1** |
| v1-BPE | 0.110 | 0.071 | 0.589 | 0 |

The single flag is the greedy run on the nonsense prompt `?!3 $&',-.:;`
(rep3 > 0.5), the same prompt that trips the shipped checkpoint. Greedy sample
for `ROMEO:` from the new checkpoint:

```
This is the bosom of the house of York
And the first of Lcaster and his crown
And the Duke of York and Lcaster Grey
Within the Duke of York and Lanaster
To the Duke of York, Somerset, Somerset,
And the Duke of York, Hen
```

Shipped v2 on the same prompt produces a shorter, less repetitive but less
coherent continuation (`Then,,, I, I'll be my wife's s.` ...). Generation
quality remains below v1-BPE; the repetition aggregate is identical within
noise.

## Reproducibility

- Two independent CPU evaluation runs (`veylith/evaluation/p1-3-results.json`
  and a second run in `/tmp`) produced identical `evaluation`, `router` and
  `generation` entries for all six checkpoints, including the new one. The
  evaluator is deterministic.
- The training run was executed once, as approved. No second run, no retry.
  Bitwise reproducibility of the GPU training path is not claimed: HIP kernels
  and the dense-combined MoE path were not tested for determinism.
- The real resume state passes `load_train_state` with the run's exact args
  (config fingerprint and section hash match), and the artifact weights are
  bit-identical to the state's `best_model`, so the saved checkpoint is the
  step-500 selection.
- The run artifacts (command, env, log, curve, checkpoint, state) live in
  `~/veylith-runs/p1-3-20261026/`. The curve is committed as
  `veylith/evaluation/p1-3-curve.jsonl`; the 102 MB checkpoint is not in git.

## Limitations

- One run, one seed. No estimate of seed variance; the -0.024 nats/token
  against the shipped checkpoint is within the paired section noise and may be
  trajectory luck.
- The selection set is six 512-character sections (1395 tokens) and the test
  set is six sections (1377 tokens). Both are small samples; per-section
  differences range from -0.100 to +0.091 nats/token against the shipped v2.
- The shipped v2 checkpoint stores no step or metrics, so "step 500" for the
  old model is documentation, not artifact evidence. Its selection criterion
  was composite; the new one is main CE. The two runs also differ in evaluation
  mode (legacy random windows vs fixed sections), which changes the RNG stream.
- Early stopping with patience was optional and is not implemented; the
  main-CE selection already keeps the early minimum, but a patience rule would
  have saved about 4000 steps.
- The run measures one corpus. Whether the gap to v1 is capacity, data volume
  or optimization is not decided by this run; the audit's parameter accounting
  and the curve point to capacity/data mismatch as the leading hypothesis.

## Next step

E2 is the highest-value controlled experiment: the same protocol and budget
with `--n_embd 192` (14.51M total, 5.67M active, head_dim 24, expert width
512), one variable changed, main-CE selection on the same sections. Given the
strong overfitting at 25.6M parameters and 0.44M training tokens, a smaller
model should fit less and may generalize better. E3 (`--full_attn_every 1`)
targets a different hypothesis (only layer 3 is full attention) and should
wait until the capacity arm is measured. A data-scaling arm would separate
capacity from corpus size, but it changes the comparison basis with v1 and
needs its own baseline decision.
