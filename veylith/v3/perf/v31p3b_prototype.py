"""v3.1-PERF-3B prototype harness: correctness gate and bounded profile.

Modes:
    correctness  baseline twice + variant once, two updates each, bitwise digests
    profile      5 warmup + 5 profiled updates for one variant, Chrome trace

No production file is modified; the variant is applied at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "veylith/v3")
sys.path.insert(0, os.path.join("veylith/v3", "perf"))

SCRATCH = os.path.expanduser("~/veylith-perf3b-20261027")
DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
STATE0 = os.path.expanduser("~/veylith-v31l-dense-20261027/ckpt_state_step0.pt")
EXPECTED_FINGERPRINT = "1212c3d134911dee2745d5deb2b91454296eaf86954d21013d4d641c0aefd260"


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def digest_tensors(items):
    hasher = hashlib.sha256()
    for name, tensor in items:
        data = tensor.detach().cpu().contiguous().float().numpy().tobytes()
        hasher.update(name.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(data)
    return hasher.hexdigest()


def model_digest(model):
    return digest_tensors(sorted(model.state_dict().items()))


def grad_digest(model):
    return digest_tensors([(name, param.grad)
                           for name, param in model.named_parameters()
                           if param.grad is not None])


def optimizer_digest(optimizer):
    state = optimizer.state_dict()["state"]
    items = []
    for index in sorted(state):
        for key in sorted(state[index]):
            value = state[index][key]
            if torch.is_tensor(value):
                items.append((f"{index}.{key}", value))
    return digest_tensors(items)


def build(variant: bool):
    import torch
    from architecture.config import get_candidate
    from architecture.model import GPT
    from training.checkpoint import restore_rng
    from training.config import TrainConfig, sha256_file
    from training.data import iter_train_batches
    from training.schedule import CosineSchedule
    from training.train import _param_groups, set_seed

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model_cfg = get_candidate("A")
    model_cfg.attention_impl = "sdpa"
    model_cfg.moe_impl = "dense"
    cfg = TrainConfig(
        data_root=DATA, out_dir=SCRATCH, model=model_cfg, device="cuda",
        precision="bf16", batch_size=8, grad_accum=8, seq_len=512,
        max_steps=500, lr=3e-4, min_lr_ratio=0.1, warmup_steps=15,
        weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=1.0,
        seed=0, eval_interval=1000, train_gap_batches=8, save_every=1000,
        select_metric="macro_main_ce", deterministic=True, exact_chars=True,
        max_wall_seconds=300.0)
    cfg.validate()
    data_sha = sha256_file(os.path.join(DATA, "corpus", "manifest.json"))
    fingerprint = cfg.fingerprint(data_sha)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(f"fingerprint mismatch: {fingerprint}")

    set_seed(cfg.seed)
    model = GPT(cfg.model).to("cuda")
    state = torch.load(STATE0, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.weight_decay),
                                  lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  eps=cfg.eps)
    scheduler = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps,
                               cfg.min_lr_ratio)
    patched = None
    if variant:
        from v31p3b_variant import apply_variant
        patched = apply_variant(model)
    restore_rng(state["rng"])
    del state
    _plan, _bpe, iterator = iter_train_batches(DATA, cfg, 0, 0, epochs=1)
    return model, optimizer, scheduler, iterator, fingerprint, patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "profile"), required=True)
    parser.add_argument("--variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    from training.precision import autocast_context
    from training.train import _loss_terms

    auxiliary_weight = 0.01

    def run_updates(model, optimizer, scheduler, iterator, count, capture=False,
                    annotate=False, head_capture=None):
        records = []
        for _ in range(count):
            def maybe_local(name):
                return record_function(name) if annotate else _null()

            data_started = time.perf_counter()
            with maybe_local("data_fetch"):
                window = [next(iterator) for _ in range(8)]
            data_seconds = time.perf_counter() - data_started
            total_valid = sum(max(0, int(length) - 1)
                              for batch in window for length in batch["lengths"])
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            cpu_started = time.process_time()
            with maybe_local("update"):
                optimizer.zero_grad(set_to_none=True)
                ce_sum_total = aux_sum_total = 0.0
                router_hash = hashlib.sha256()
                with autocast_context("cuda", "bf16"):
                    for batch in window:
                        with maybe_local("forward"):
                            ce_sum, aux, _count, router = _loss_terms(
                                model, batch, "cuda", auxiliary_weight)
                            loss = (ce_sum + auxiliary_weight * aux * _count) / total_valid
                            if not math.isfinite(float(loss.detach())):
                                raise RuntimeError("non-finite loss")
                        with maybe_local("backward"):
                            loss.backward()
                        ce_sum_total += float(ce_sum.detach())
                        aux_sum_total += float(aux.detach()) * _count
                        for layer, top in enumerate(router):
                            router_hash.update(layer.to_bytes(2, "little"))
                            router_hash.update(top.detach().cpu().numpy().tobytes())
                with maybe_local("grad_clip"):
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0))
                if not math.isfinite(grad_norm):
                    raise RuntimeError("non-finite gradient norm")
                with maybe_local("optimizer"):
                    lr = scheduler.advance()
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            record = {
                "main_ce": ce_sum_total / total_valid,
                "aux_sum": aux_sum_total / total_valid,
                "loss": ce_sum_total / total_valid
                + auxiliary_weight * aux_sum_total / total_valid,
                "grad_norm": grad_norm,
                "lr": lr,
                "step_seconds": step_seconds,
                "data_seconds": data_seconds,
                "process_cpu_seconds": time.process_time() - cpu_started,
                "router_digest": router_hash.hexdigest(),
            }
            if capture:
                record["model_digest"] = model_digest(model)
                record["grad_digest"] = grad_digest(model)
                record["optimizer_digest"] = optimizer_digest(optimizer)
            records.append(record)
            if head_capture is not None and "logits_digest" not in head_capture:
                head_capture["logits_digest"] = digest_tensors(
                    [("head", head_capture["logits"])])
                del head_capture["logits"]
        return records

    if args.mode == "correctness":
        results = {}
        os.makedirs(SCRATCH, exist_ok=True)
        for label, variant in (("baseline_a", False), ("baseline_b", False),
                               ("variant", True)):
            model, optimizer, scheduler, iterator, fingerprint, patched = build(variant)
            captured = {}

            def capture_head(module, inputs, output):
                if "logits_digest" not in captured and "logits" not in captured:
                    captured["logits"] = output.detach()

            handle = model.head.register_forward_hook(capture_head)
            try:
                records = run_updates(model, optimizer, scheduler, iterator, 2,
                                      capture=True, annotate=False,
                                      head_capture=captured)
            finally:
                handle.remove()
            results[label] = {
                "fingerprint": fingerprint,
                "patched": patched,
                "records": records,
                "logits_digest": captured.get("logits_digest"),
            }
            del model, optimizer
            torch.cuda.empty_cache()

        def differences(a, b):
            keys = ("main_ce", "aux_sum", "loss", "grad_norm", "lr",
                    "router_digest", "model_digest", "grad_digest",
                    "optimizer_digest")
            out = []
            for index in range(len(a["records"])):
                for key in keys:
                    if a["records"][index][key] != b["records"][index][key]:
                        out.append({"update": index + 1, "field": key,
                                    "baseline": a["records"][index][key]
                                    if key != "router_digest" else "digest",
                                    "variant": b["records"][index][key]
                                    if key != "router_digest" else "digest"})
            if a["logits_digest"] != b["logits_digest"]:
                out.append({"update": 1, "field": "logits_digest"})
            return out

        report = {
            "mode": "correctness",
            "results": results,
            "baseline_a_vs_baseline_b": differences(results["baseline_a"],
                                                    results["baseline_b"]),
            "baseline_vs_variant": differences(results["baseline_a"],
                                               results["variant"]),
        }
        report["deterministic_baseline"] = not report["baseline_a_vs_baseline_b"]
        report["variant_bitwise_equal"] = not report["baseline_vs_variant"]
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("correctness deterministic", report["deterministic_baseline"],
              "variant equal", report["variant_bitwise_equal"])
        return 0 if (report["deterministic_baseline"]
                     and report["variant_bitwise_equal"]) else 1

    # profile mode
    started = time.monotonic()
    model, optimizer, scheduler, iterator, fingerprint, patched = build(bool(args.variant))
    warmup = run_updates(model, optimizer, scheduler, iterator, 5, annotate=False)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"trace_variant{args.variant}.json")
    exported = {"called": False}

    def on_trace(prof):
        prof.export_chrome_trace(trace_path)
        exported["called"] = True

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=False, with_stack=False,
        schedule=schedule(wait=0, warmup=0, active=5, repeat=1),
        on_trace_ready=on_trace)
    measured = []
    with prof:
        for _ in range(5):
            measured.extend(run_updates(model, optimizer, scheduler, iterator, 1,
                                        annotate=True))
            prof.step()
    tables = {}
    for label, sort_key in (("cuda", "self_cuda_time_total"),
                            ("cpu", "self_cpu_time_total")):
        try:
            tables[label] = prof.key_averages().table(
                sort_by=sort_key, row_limit=60, max_name_column_width=120)
        except Exception as exc:
            tables[label] = f"table failed: {exc!r}"
        with open(os.path.join(out_dir, f"key_averages_{label}_variant{args.variant}.txt"),
                  "w", encoding="utf-8") as handle:
            handle.write(tables[label])

    payload = {
        "mode": "profile",
        "variant": bool(args.variant),
        "patched": patched,
        "fingerprint": fingerprint,
        "warmup": warmup,
        "measured": measured,
        "trace_path": trace_path,
        "trace_exported": exported["called"],
        "trace_bytes": os.path.getsize(trace_path) if os.path.exists(trace_path) else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.monotonic() - started,
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("profile variant", args.variant, "exported", exported["called"],
          "trace", payload["trace_bytes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
