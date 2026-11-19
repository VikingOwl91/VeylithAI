# One E16/K4 training step on CPU

VEYLITH-6 tests whether Rust and the archived Python MoE agree on a small synthetic training step. Both Rust execution paths selected the same four experts, in the same order, as Python for each of 36 inputs. All 16 experts received tokens. Each comparison covered every element of 54 parameter tensors, including the embedding, router, experts, shared expert and output projection.

| Against archived Python, float32 CPU | Rust dense-combined | Rust four selected experts per token |
|---|---:|---:|
| Next-token cross-entropy | 3.463654995 | 3.463654995 |
| Training objective including router loss | 3.473684311 | 3.473684311 |
| Largest absolute error across CE, router loss and objective | 0 | 0 |
| Largest absolute gradient error | 2.79e-9 | 1.86e-9 |
| Largest absolute error in the AdamW parameter change | 7.45e-9 | 2.98e-8 |

The gate requires exact ordered top-4 choices, and absolute errors at most 1e-6 for the losses, 1e-6 for each gradient, and 2e-6 for each parameter change. The second Rust implementation visits four experts per token in a CPU loop. These measurements establish arithmetic agreement for this one step. GPU stability, BF16 behavior, and training throughput are separate measurements.

## Reproduce

From the Git repository root with the [Rust build environment](../../driver/README.md#run-a-sample), select a new report filename.

```sh
cargo run --release --locked --bin e16-parity -- \
  --fixture veylith/driver/fixtures/e16-k4-cpu.npz --out e16-parity.json
```

The [synthetic test vector](../../driver/fixtures/e16-k4-cpu.npz) stores inputs and independent expected outputs. The binary checks its SHA-256 digest, runs with one CPU thread and publishes a report when both comparisons pass. [Recorded report](e16-k4-cpu-parity.json)

## Reference calculation

The generator loaded `Config` and `MoE` from the archived v2 trainer at blob `bcd5eb60524a5ec15089fd9fdfca09e3a2761e8f`. Python PyTorch 2.15.0 and Rust tch 0.27.0 used the workstation's LibTorch 2.15.0 CPU backend. [Archived class](https://github.com/VikingOwl91/VeylithAI/blob/be68c2c32e9629a14a5a6ae8329ca214ea639044/veylith/v2/train.py#L368)

1. Use a vocabulary of 31, embedding width 8, expert hidden width 16, 16 routed experts, top-4, one shared expert, and zero dropout. Add a bias-free, untied output projection. The parameter prefixes are `embed`, `moe`, and `head`.
2. Sort the fully qualified parameter names lexicographically. Use one float32 CPU PyTorch generator with seed 20261119 and assign `(rand(shape) - 0.5) * 0.5` to each parameter in that order. The test vector stores these initial values explicitly.
3. Generate four ten-token rows. For row `b` and column `i`, the ID is `(7 * (9*b + i) + 3) % 31`. Embed the first nine IDs in each row and predict the next nine. Use the archived MoE's dense-combined output plus the input embedding as the input to the projection. The objective is mean next-token cross-entropy plus 0.01 times the archived router load-balancing loss.
4. Save the ordered top-4 IDs and three losses. Backpropagate once. Save every gradient and each parameter's change after one AdamW step with learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, and PyTorch's `foreach=False`.
