"""Measure one arm of the Phase 4 eight-expert batch-delivery comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

import sparse_moe_v2_train as sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))
import train as dense


BATCH = 64
BLOCK = 256
MODEL_SEED = 1337
DATA_SEED = 1338


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--impl", choices=("sparse", "dense"), required=True)
    parser.add_argument("--delivery", choices=("normal", "prestaged"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    total = args.warmup + args.steps
    if args.warmup < 1 or args.steps < 1 or total > 1500:
        parser.error("warmup and steps must be positive and total at most 1500")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("this experiment requires a visible bf16-capable GPU")

    impl = {"sparse": sparse, "dense": dense}[args.impl]
    with args.ids.open("rb") as stream:
        corpus_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    data = torch.load(args.ids, map_location="cpu", weights_only=True)
    if data.ndim != 1 or data.dtype != torch.long or data.min() < 0 or data.max() >= 1000:
        parser.error("expected one-dimensional BPE-1000 token IDs of dtype long")
    train = data[:int(0.9 * len(data))]
    if len(train) <= BLOCK:
        parser.error("training split is too short")

    torch.manual_seed(MODEL_SEED)
    model = impl.GPT(impl.Config(vocab_size=1000)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Both production loaders sample CPU window offsets once per batch. Verify
    # their stream across warmup and measurement outside the timing window.
    offsets = torch.Generator().manual_seed(DATA_SEED)
    digest = hashlib.sha256()
    for _ in range(total):
        starts = torch.randint(len(train) - BLOCK, (BATCH,), generator=offsets)
        digest.update(bytes(starts.view(torch.uint8).tolist()))
    torch.random.default_generator.manual_seed(DATA_SEED)
    staged = []
    staging_seconds = 0.0
    if args.delivery == "prestaged":
        torch.cuda.synchronize()
        start = perf_counter()
        staged = [impl.get_batch(train, BLOCK, BATCH, "cuda") for _ in range(total)]
        torch.cuda.synchronize()
        staging_seconds = perf_counter() - start

    delivery_seconds = 0.0
    for step in range(total):
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            window_start = perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = impl.lr_at(step, 1500, 100, 3e-4)
        start = perf_counter()
        x, y = staged[step] if staged else impl.get_batch(train, BLOCK, BATCH, "cuda")
        elapsed = perf_counter() - start
        if step >= args.warmup:
            delivery_seconds += elapsed
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    window = perf_counter() - window_start
    if not torch.equal(torch.get_rng_state(), offsets.get_state()):
        raise RuntimeError("CPU random stream diverged from expected batch offsets")
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite final loss")
    print(json.dumps(dict(impl=args.impl, delivery=args.delivery, warmup=args.warmup,
                          steps=args.steps, torch_version=torch.__version__, hip=torch.version.hip,
                          gpu=torch.cuda.get_device_name(), corpus_sha256=corpus_digest,
                          offsets_sha256=digest.hexdigest(), model_seed=MODEL_SEED, data_seed=DATA_SEED,
                          staging_seconds=staging_seconds, measured_seconds=window,
                          tokens_per_second=BATCH * BLOCK * args.steps / window,
                          delivery_ms_per_step=1000 * delivery_seconds / args.steps,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                          final_loss=loss.item())))


if __name__ == "__main__":
    main()
