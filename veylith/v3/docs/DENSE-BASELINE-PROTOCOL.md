# VEYLITH-8 · Small dense Transformer baseline

The primary measurement is the change in full-validation, source-macro next-token cross-entropy (CE) from initialization to update 500. The comparison point is the v3.1-L dense-combined MoE run on the frozen multi-source corpus. That run reached 5.40009 nats/target from an initial 9.11677. [Reference run](dense-run-v31l-2026-10-27-summary.json)

## Results on 19 November 2026

The small dense Transformer reached **6.90331 nats per target**, down from 9.02217. It saw the same 16,352,000 ordered training targets and all 2,118,230 validation targets as the v3.1-L comparison. The MoE's loss at update 500 was 1.50322 nats lower on the source-macro measure. Model size, depth, precision and training compute also differed. This run compares the two complete configurations at the same token budget. [Dense receipt](dense-baseline-500.json) · [MoE receipt](dense-run-v31l-2026-10-27-summary.json) · [Corpus origin and terms](../../TRAINING-DATA.md)

| Update-500 mean CE | Validation targets | Small dense | v3.1-L MoE |
|---|---:|---:|---:|
| Cosmopedia v2 | 310,531 | 7.02791 | 5.34397 |
| FineWeb-Edu | 1,296,602 | 6.82246 | 5.44800 |
| Wikipedia | 511,097 | 6.85956 | 5.40831 |
| Equal weight per source | 2,118,230 | **6.90331** | **5.40009** |
| Equal weight per target | 2,118,230 | 6.86153 | 5.42317 |

The RX 9070 XT run saved update 1 and resumed to 500 with the same weights, Adam state, training order and software fingerprint. Training took 96.43 seconds, or 169,580 targets/s. The two invocations used 108.95 active wall seconds including both full validations. The receipt includes SHA-256 hashes for the saved checkpoints at updates 1 and 500.

The verified HIP stream priority was +1, the lowest on this device, and the device-wide mask read back as `0xff`. The trainer took 21,156 synchronized HIP memory samples with a minimum of 15.14 GiB free. An independent Linux DRM reading during training showed 12.81 GiB free. Both readings exceeded the requested 6 GiB floor. The DRM reading is a single workstation-wide observation.

An initial startup check stopped before model allocation because the nondefault-stream mask query left its output untouched. A separate queue diagnostic submitted no model work and exited with a userspace assertion. The completed run verified the device-wide mask and recorded the nondefault-stream readback limitation in both the checkpoint and the receipt.

## Fixed experiment

| Quantity | Small dense baseline |
|---|---|
| Model | One pre-LayerNorm decoder block, four full causal attention heads, width 64, GELU MLP width 256, learned positions, untied output embedding |
| Parameters | 1,130,880 including position embeddings and LayerNorm. The v3.1-L reference had 98,340,864. |
| Precision and initialization | FP32, seed 0, zero dropout. Embedding and projection weights follow a zero-mean normal distribution with standard deviation 0.02. |
| Data | Frozen multi-100m corpus and its 8,192-piece tokenizer. The receipt records hashes for the ordered training sequences and validation sources. |
| Training | 500 updates of 64 sequences, each with 512 token IDs and 511 next-token targets. The optimizer sees 16,352,000 targets. |
| Optimizer | AdamW, betas 0.9 and 0.999, epsilon 1e-8, weight decay 0.01 on matrices, global gradient norm clip 1.0. Linear warmup for 15 updates to 0.0003, then cosine decay toward 10% of that rate |
| Validation | At initialization and update 500, score 310,531 Cosmopedia v2 targets, 1,296,602 FineWeb-Edu targets and 511,097 Wikipedia targets, excluding padding. Report each source's mean CE, the arithmetic mean of those three means, and their target-weighted mean. |
| Compute proxy | 3,833,088 training GEMM FLOPs per target, about 62.68 trillion over the run. Uses three times forward GEMM cost for forward and backward, counts full-square QK and AV, and omits embedding lookup, normalization, softmax and optimizer work. Observed time and target throughput are reported separately |

The comparison controls the tokenizer, corpus, optimizer target budget and evaluation splits. Parameter count, architecture, precision and compute differ. A CE difference alone cannot attribute a benefit to MoE routing. It characterizes these two trained models at the stated token horizon.

## Commands

From the repository root, set `DATA` to the frozen multi-100m corpus directory and `RUN` to a new directory for the small model. The runner defaults to CPU, lowers its priority to nice 15 and uses two tensor threads and one interop thread. The four-row default microbatch yields sixteen backwards per optimizer update.

```sh
export CARGO_TARGET_DIR="$PWD/.codex/target" LIBTORCH=/usr CXX=g++
cargo build -j 2 --release --locked --bin dense-baseline
nice -n 15 taskset -c 4,5 "$CARGO_TARGET_DIR/release/dense-baseline" \
  --data-root "$DATA" --run-dir "$RUN" --microbatch 4 --save-every 25
```

For an authorized dense ROCm run, build with `--features rocm`, use `--device rocm` and apply the approved eight-bit HIP compute mask `ROCR_VISIBLE_DEVICES=0 ROC_GLOBAL_CU_MASK=0xff` before starting the process. The binary checks the device visibility and mask environment, requires exactly one HIP device at index zero, requests and verifies the lowest HIP stream priority, and requires a HIP device-wide mask readback of 0xff. The training stream is created without a custom mask. The backend applies the device-wide mask at queue creation and fails queue creation if applying the mask fails. In WGP mode it expands a mask bit to a pair of physical CUs. [ROCm queue implementation](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.4/projects/clr/rocclr/device/rocm/rocdevice.cpp#L2958-L3023)

Where HIP returns a stream mask, the trainer checks it. On ROCm 7.2.4 the nondefault-stream query can leave its output buffer untouched when the global mask is active. The trainer queries two buffers with different initial contents to identify this case and records it in the receipt. [HIP stream implementation](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.4/projects/clr/hipamd/src/hip_stream.cpp#L795-L836)

FP32 output logits dominate this model's memory traffic. A smaller microbatch reduces peak memory and increases the number of backward passes per update. Resume checks the microbatch setting recorded in the checkpoint. [Native ROCm protocol](../../driver/README.md#run-on-rocm)

## Checkpoint and completion

An atomic `latest.ot` holds named parameters, both Adam moments, optimizer update, consumed targets, schedule position (derived from the update), initial validation, timing and corpus/protocol identity. A run-directory lock permits one writer. Checkpoints are saved after initial validation, every configured interval and at the requested stop step. Use `--stop-after N` for an intermediate milestone and `--resume` with the same corpus, backend, microbatch and code to continue to 500. Incomplete milestones leave the final result path available for completion. A restart after the step-500 checkpoint can perform final validation and publish the result.

Successful completion writes `result.json` with initial and final validation, parameters, data evidence, train CE samples, timings, the GEMM proxy and actual target count. Training throughput covers synchronized optimizer updates. Active wall seconds sum process segments and exclude any delay between runs. Interrupted work after the last durable checkpoint is excluded. CPU optimizer parity is exercised by `cargo test --locked --bin dense-baseline causal_prefix_and_resumed_adam_step_match_uninterrupted_cpu`, which checks causal prefix invariance and bit-exact weights and both moments one update after reloading a tiny checkpoint.

## Device headroom and provenance

ROCm runs require 6 GiB free device VRAM. The trainer samples HIP free and total memory at startup, before allocating the model, before and after each training and validation microbatch, and around every optimizer update. Admission to an allocation requires a further 2 GiB of slack (8 GiB free). Before each sample the trainer synchronizes its device work. The final receipt reports sample count, minimum, maximum and total bytes. Another process can allocate memory between samples.

A failed memory probe or threshold check writes `vram-stop.json` and exits. The file records the phase, threshold, observed availability, observed range and last confirmed durable update in this invocation. An interrupted gradient accumulation is discarded. After checking headroom, use `--resume` when `latest.ot` exists. A stop before the initial checkpoint can start again without that flag.

The checkpoint protocol includes SHA-256 identities of the dense CLI, model, data loader, native ROCm bridge, Cargo lockfile, both manifests and build script. It pins the ROCm feature, loaded LibTorch/ROCm library identities (ELF build ID, or file SHA-256), and for ROCm the HIP runtime and driver versions, GPU name, architecture and PCI bus ID. A resume with a different recorded protocol is rejected.
