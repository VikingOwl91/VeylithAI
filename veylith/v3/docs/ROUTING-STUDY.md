# Frozen-checkpoint routing study

VEYLITH-11 measures next-token cross-entropy (CE, nats per token) on the same 240 validation sequences as the [expert-behavior analysis](V3.1-N-EXPERT-BEHAVIOR.md). VEYLITH-9 adds a one-layer routing intervention. The reference uses update 500 of the v3.1-L dense-combined model in CPU float32 evaluation mode. It scores 511 next-token targets per sequence and excludes padding and the auxiliary routing loss.

From the repository root, set `DATA` to the multi-100m corpus directory and `CKPT` to the saved `ckpt_state_step500.pt`. Choose a new output filename for each run.

```sh
nice -n 15 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python veylith/v3/analysis/routing_study.py --data-root "$DATA" \
  --checkpoint "$CKPT" --threads 2 --out baseline.json
```

Add `--ablate` for all seven MoE blocks and three fixed permutation seeds (11, 29, 47). Use `--layers 1` for the first MoE block or `--seeds 11` for one seed. Indices are zero-based absolute transformer block indices. The CLI records completed runs only. It verifies the checkpoint and tokenizer file hashes, corpus manifest bytes and ordered validation token digest against the committed historical records before inference. It reads eight sequences at a time, in ten batches per source. Saved reports contain aggregate metrics and identifiers. The corpus and checkpoint stay read-only.

## Interpretation

For each layer, the evaluator caches its ordinary input and replays the model suffix. The identity intervention must reproduce every valid baseline token CE exactly. The shuffle moves routing choices and their weights together across valid positions in each source-pure batch. Other layers run their usual router on the resulting hidden states. This offline perturbation can use donor choices from later positions. It measures sensitivity to breaking the learned association in the frozen model.

Classes describe the input token at position t, whose output predicts t+1. Punctuation-only and sentence-end categories can overlap. The table includes each class's token and sequence counts. Positive delta CE means shuffling worsened prediction. Each 95% percentile interval resamples paired sequences within source (1,000 draws). The mean over fixed seeds averages the per-sequence deltas before resampling. These exploratory intervals describe variation across the sampled sequences conditional on the chosen permutations. They do not account for repeated class and layer comparisons or variation across checkpoints. The reference routing analysis recommended a separate, class-targeted intervention. This experiment perturbs all valid positions in one layer and reports class-conditioned effects.

## Sentence-ending punctuation in the first expert layer

The follow-up uses two shuffles of block 1 on the update-500 checkpoint. The unrestricted shuffle can move any valid routing row to any valid position in the same batch. The second shuffle uses the same random donor order, retaining only donors from the recipient's group. One group consists of input token pieces containing a period, exclamation mark, or question mark. The other contains all remaining valid positions. Both groups are shuffled. A period in an abbreviation or number also meets the rule.

The primary outcome is the difference in next-token CE at positions in the punctuation group, computed as unrestricted minus group-preserving. A positive value measures the loss recovered by preserving group membership. Pair the two arms on the same tokens, average the paired differences over seeds 11, 29 and 47, then bootstrap whole eight-sequence batches within each corpus source (1,000 draws). Keeping the batch intact accounts for shared donors. Later tokens in a sequence can also respond to routing changes at earlier positions.

From the repository root, with the same `DATA` and `CKPT` as above:

```sh
nice -n 15 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python veylith/v3/analysis/routing_study.py --data-root "$DATA" \
  --checkpoint "$CKPT" --threads 2 --ablate --layers 1 --seeds 11 29 47 \
  --compare-sentence-end --out sentence-end.json
```

The first 240 sequences informed the choice of layer and class. The prespecified follow-up rule is to repeat the comparison if the primary 95% interval is strictly above zero. Add `--fresh-sample` and choose a new output path to score the next ten batches per source. The loader verifies the original sample digest and rejects overlap in full-sequence content. Adjacent windows can come from the same document.

### Results at update 500

Preserving the punctuation group recovered about 95% of the extra next-token loss caused by unrestricted shuffling at those positions. The result repeated on the reserved validation sequences. Each sample had 240 sequences and 122,640 valid next-token targets.

| Validation sample | Qualifying positions | Unrestricted penalty | Group-preserving penalty | Recovered CE (95% paired interval) |
|---|---:|---:|---:|---:|
| [Original](routing-sentence-end-update500.json) | 4,228 | 0.14190 | 0.00756 | 0.13434 (0.11677–0.15062) |
| [Confirmation](routing-sentence-end-update500-confirmation.json) | 4,335 | 0.16594 | 0.00771 | 0.15823 (0.13277–0.18323) |

Both penalties are increases over normal routing, in nats per qualifying target. All three seeds gave positive recovery on each sample. On the confirmation sample, preserving the grouping recovered 0.00946 nats per target across **all** positions, about 9.6% of the unrestricted penalty overall. The grouping explains much more of the measured penalty immediately after these pieces than across the sample as a whole.

The identity control reproduced each valid token's baseline CE exactly on both samples. On the original sample, the baseline and the unrestricted per-seed measurements matched the previous study exactly. The run measures the effect of preserving the grouping in block 1 of this checkpoint. It does not identify a particular expert as responsible or establish that the model recognizes grammatical sentence boundaries.

## Separating expert choice from mixture weights

The block-1 follow-up crosses two components of the top-2 route. At a given position, one donor supplies the ranked pair of expert IDs. Another supplies the weights on the first and second ranks. Each batch and seed uses exactly the unrestricted and group-preserving donor permutations from the preceding experiment.

| Condition | Ranked expert IDs | Weight on each rank |
|---|---|---|
| Unrestricted | Unrestricted donor | Same unrestricted donor |
| Preserve IDs only | Same-group donor | Unrestricted donor |
| Preserve weights only | Unrestricted donor | Same-group donor |
| Preserve both | Same-group donor | Same donor as IDs |

All four interventions preserve how often each expert is selected across the batch and the batch-wide distribution of ranked weight pairs. Crossing components can change the total weight assigned to an individual expert. The evaluator constructs gate scores that reproduce the desired pair and its two weights, then scores next-token CE. The auxiliary load-balancing loss is outside this comparison.

The primary outcome is recovery at input pieces containing `.`, `!`, or `?`: subtract each single-component condition's CE from unrestricted CE. Directly compare the two recoveries on identical batches. The interaction is the recovery with both components preserved minus the sum of the two individual recoveries. A positive interaction means joint preservation recovered more than the sum. Intervals resample whole batches within source and condition on the same three fixed seeds as above.

Add `--cross-components` to the block-1 sentence-end command and use a new output filename. Use `--fresh-sample` for the second sample. These are the two samples from the preceding study, analyzed separately.

### Crossed results at update 500

Preserving only the ranked expert IDs recovered 97.6% of the joint recovery on the original sample and 95.1% on the second sample. All three seeds on each sample favored preserving IDs. Preserving the weight split alone made the unrestricted assignments slightly worse.

| Sample | Qualifying positions | Preserve IDs only | Preserve weights only | Preserve both | IDs-only advantage over weights-only (95% paired interval) |
|---|---:|---:|---:|---:|---:|
| [Original](routing-components-update500.json) | 4,228 | +0.13114 | −0.01118 | +0.13434 | +0.14232 (0.12617–0.15759) |
| [Second sample](routing-components-update500-confirmation.json) | 4,335 | +0.15047 | −0.01105 | +0.15823 | +0.16152 (0.13983–0.18386) |

Values are recovered next-token CE in nats per qualifying position, relative to unrestricted shuffling. On the second sample, preserving the weight split in addition to the ranked IDs recovered another 0.00776 nats. The interaction was +0.01881 (95% interval 0.01242–0.02549), reflecting how the value of keeping the weight split depends on which experts receive it.

The baseline, identity control, and both previous shuffle conditions reproduced the earlier measurements exactly on each sample. These results establish the importance of the ranked pair in this block at this checkpoint. Isolating expert 3's contribution would require varying its assignment separately.
