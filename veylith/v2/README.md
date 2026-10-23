# veylith v2

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
python train.py                                           # char tokenizer
python train.py --tokenizer bpe --bpe_vocab_size 1000     # custom BPE
python sample.py                                          # generate from checkpoint
```

## Notes

- MoE training uses the classic Switch-style auxiliary load-balancing
  loss. The frontier models in `../reference/` use aux-loss-free
  balancing instead (`topk_method: "noaux_tc"` in their configs).
- The MoE is dense-combined: routing is top-2, but every expert computes
  on every token and the router weights combine the results. True
  sparse routing needs variable-size indexing (gather/scatter with
  per-batch sizes), which faults intermittently on ROCm. Dense-combined
  costs E/k times the expert FLOPs and avoids that entirely. Frontier
  labs use custom grouped-GEMM kernels for sparse routing.
- MTP here is a training-time loss (position t also predicts t+2).
  The inference-time payoff, speculative decoding, is a serving trick
  and is not wired into `sample.py`.
- The first `--first_k_dense` layers are dense, the rest MoE, like
  GLM-5's `first_k_dense_replace`.
- Early stopping: the saved checkpoint holds the best-validation
  weights, not the final ones.
- The reported loss includes the MTP term and the MoE aux term (small).
