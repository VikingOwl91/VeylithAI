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
