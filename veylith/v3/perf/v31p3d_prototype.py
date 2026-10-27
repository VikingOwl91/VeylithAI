"""v3.1-PERF-3D prototype harness: operator gate, training gate, bounded profile.

Modes:
    gate1         bitwise operator comparison of the fused SwiGLU forward/backward
    correctness   baseline twice + variant once, two updates, bitwise digests
    profile       warmup plus measured updates for one variant, Chrome trace

The production model is not modified; the SwiGLU forward is patched at runtime.
Timing runs keep one device synchronization per update (checks and router
digest are batched), so wall time is not inflated by per-micro-batch stalls.
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

SCRATCH = os.path.expanduser("~/veylith-perf3d-20261027")
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


def bits(tensor):
    return tensor.contiguous().view(torch.int16)


def gate1():
    import torch.nn.functional as F

    from v31p3d_swiglu import FusedSiluMul, fused_silu_mul

    torch.manual_seed(20261027)
    cases = []

    def compare(name, a, b):
        t_ref = F.silu(a)
        y_ref = t_ref * b
        y, t = fused_silu_mul(a, b)
        cases.append({
            "name": name,
            "shape": list(a.shape),
            "t_bitwise": bool(torch.equal(bits(t), bits(t_ref))),
            "y_bitwise": bool(torch.equal(bits(y), bits(y_ref))),
            "t_max_abs": float((t.float() - t_ref.float()).abs().max()),
            "y_max_abs": float((y.float() - y_ref.float()).abs().max()),
        })

    for shape in ((4088, 960), (1, 960), (17, 960), (4097, 961)):
        a = (torch.randn(*shape) * 4).to(torch.bfloat16).cuda()
        b = (torch.randn(*shape) * 2).to(torch.bfloat16).cuda()
        compare(str(shape), a, b)
    zeros = torch.zeros(4088, 960, dtype=torch.bfloat16, device="cuda")
    compare("zeros", zeros, zeros.clone())
    compare("large_positive",
            torch.full((4088, 960), 50.0, dtype=torch.bfloat16, device="cuda"),
            torch.ones(4088, 960, dtype=torch.bfloat16, device="cuda"))
    compare("large_negative",
            torch.full((4088, 960), -50.0, dtype=torch.bfloat16, device="cuda"),
            torch.ones(4088, 960, dtype=torch.bfloat16, device="cuda"))
    compare("small_1e-30",
            torch.full((4088, 960), 1e-30, dtype=torch.bfloat16, device="cuda"),
            torch.ones(4088, 960, dtype=torch.bfloat16, device="cuda"))
    one = torch.ones(4088, 960, dtype=torch.bfloat16, device="cuda")
    two = torch.full((4088, 960), 2.0, dtype=torch.bfloat16, device="cuda")
    compare("bf16_boundary", torch.nextafter(one, two), one)

    a = (torch.randn(4088, 960) * 2).to(torch.bfloat16).cuda().requires_grad_()
    b = (torch.randn(4088, 960) * 2).to(torch.bfloat16).cuda().requires_grad_()
    grad = (torch.randn(4088, 960) * 0.01).to(torch.bfloat16).cuda()
    t_ref = F.silu(a)
    y_ref = t_ref * b
    y_ref.backward(grad, retain_graph=True)
    ga_ref, gb_ref = a.grad.clone(), b.grad.clone()
    a.grad = None
    b.grad = None
    y = FusedSiluMul.apply(a, b)
    y.backward(grad)
    backward = {
        "y_bitwise": bool(torch.equal(bits(y), bits(y_ref))),
        "ga_bitwise": bool(torch.equal(bits(a.grad), bits(ga_ref))),
        "gb_bitwise": bool(torch.equal(bits(b.grad), bits(gb_ref))),
        "ga_max_abs": float((a.grad.float() - ga_ref.float()).abs().max()),
        "gb_max_abs": float((b.grad.float() - gb_ref.float()).abs().max()),
    }

    rejections = {}
    try:
        square = torch.randn(960, 960, dtype=torch.bfloat16, device="cuda").t()
        fused_silu_mul(square, square)
        rejections["non_contiguous"] = "not rejected"
    except ValueError as exc:
        rejections["non_contiguous"] = str(exc)
    try:
        fused_silu_mul(torch.randn(4, 4, dtype=torch.float32, device="cuda"),
                       torch.randn(4, 4, dtype=torch.float32, device="cuda"))
        rejections["float32"] = "not rejected"
    except ValueError as exc:
        rejections["float32"] = str(exc)
    try:
        fused_silu_mul(torch.randn(4, 4, dtype=torch.bfloat16),
                       torch.randn(4, 4, dtype=torch.bfloat16))
        rejections["cpu"] = "not rejected"
    except ValueError as exc:
        rejections["cpu"] = str(exc)

    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = True
    probe = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    fused_silu_mul(probe, probe)
    fill_flag_restored = torch.utils.deterministic.fill_uninitialized_memory is True
    torch.utils.deterministic.fill_uninitialized_memory = previous_fill

    passed = all(case["t_bitwise"] and case["y_bitwise"] for case in cases)
    passed = passed and all(backward[key] for key in ("y_bitwise", "ga_bitwise", "gb_bitwise"))
    passed = passed and fill_flag_restored
    return {"cases": cases, "backward": backward, "rejections": rejections,
            "fill_flag_restored": fill_flag_restored, "passed": passed}


def build(variant: bool, max_seconds: float = 300.0):
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
        max_wall_seconds=max_seconds)
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
        from v31p3d_swiglu import apply_variant
        patched = apply_variant(model)
    restore_rng(state["rng"])
    del state
    _plan, _bpe, iterator = iter_train_batches(DATA, cfg, 0, 0, epochs=1)
    return model, optimizer, scheduler, iterator, fingerprint, patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate1", "correctness", "profile",
                                           "timing"),
                        required=True)
    parser.add_argument("--variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    from training.precision import autocast_context
    from training.train import _loss_terms

    auxiliary_weight = 0.01

    if args.mode == "gate1":
        report = gate1()
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("gate1 passed", report["passed"])
        return 0 if report["passed"] else 1

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
                ce_terms, aux_terms, loss_terms, router_parts = [], [], [], []
                with autocast_context("cuda", "bf16"):
                    for batch in window:
                        with maybe_local("forward"):
                            ce_sum, aux, count_terms, router = _loss_terms(
                                model, batch, "cuda", auxiliary_weight)
                            loss = (ce_sum + auxiliary_weight * aux * count_terms) / total_valid
                        with maybe_local("backward"):
                            loss.backward()
                        ce_terms.append(ce_sum.detach())
                        aux_terms.append(aux.detach() * count_terms)
                        loss_terms.append(loss.detach())
                        for layer, top in enumerate(router):
                            router_parts.append(top.detach().reshape(-1))
                with maybe_local("checks"):
                    ce_total = torch.stack(ce_terms).sum()
                    aux_total = torch.stack(aux_terms).sum()
                    finite = torch.isfinite(torch.stack(loss_terms)).all()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0)
                    router_flat = torch.cat(router_parts)
                    scalars = torch.cat([ce_total.reshape(1), aux_total.reshape(1),
                                         grad_norm.detach().reshape(1)]).cpu()
                    router_host = router_flat.cpu().numpy().tobytes()
                if not bool(finite.item()):
                    raise RuntimeError("non-finite loss")
                ce_sum_total = float(scalars[0])
                aux_sum_total = float(scalars[1])
                grad_norm = float(scalars[2])
                if not math.isfinite(grad_norm):
                    raise RuntimeError("non-finite gradient norm")
                with maybe_local("optimizer"):
                    lr = scheduler.advance()
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            router_hash = hashlib.sha256()
            router_hash.update(router_host)
            record = {
                "main_ce": ce_sum_total / total_valid,
                "aux_sum": aux_sum_total / total_valid,
                "loss": ce_sum_total / total_valid
                + auxiliary_weight * aux_sum_total / total_valid,
                "grad_norm": grad_norm,
                "lr": lr,
                "ce_targets": total_valid,
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
            results[label] = {"fingerprint": fingerprint, "patched": patched,
                              "records": records,
                              "logits_digest": captured.get("logits_digest")}
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
                        out.append({"update": index + 1, "field": key})
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

    started = time.monotonic()
    model, optimizer, scheduler, iterator, fingerprint, patched = build(
        bool(args.variant))
    warmup = run_updates(model, optimizer, scheduler, iterator, 5, annotate=False)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or f"variant{args.variant}"

    if args.mode == "timing":
        measured = run_updates(model, optimizer, scheduler, iterator,
                               args.updates, annotate=False)
        payload = {
            "mode": "timing",
            "variant": bool(args.variant),
            "tag": tag,
            "patched": patched,
            "fingerprint": fingerprint,
            "warmup": warmup,
            "measured": measured,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "torch": {"version": torch.__version__, "hip": torch.version.hip},
            "device": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.monotonic() - started,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("timing", tag, "updates", len(measured))
        return 0

    trace_path = os.path.join(out_dir, f"trace_{tag}.json")
    exported = {"called": False}

    def on_trace(prof):
        prof.export_chrome_trace(trace_path)
        exported["called"] = True

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=False, with_stack=False,
        schedule=schedule(wait=0, warmup=0, active=args.updates, repeat=1),
        on_trace_ready=on_trace)
    measured = []
    with prof:
        for _ in range(args.updates):
            measured.extend(run_updates(model, optimizer, scheduler, iterator, 1,
                                        annotate=True))
            prof.step()
    tables = {}
    if not args.no_tables:
        for label, sort_key in (("cuda", "self_cuda_time_total"),
                                ("cpu", "self_cpu_time_total")):
            try:
                tables[label] = prof.key_averages().table(
                    sort_by=sort_key, row_limit=60, max_name_column_width=120)
            except Exception as exc:
                tables[label] = f"table failed: {exc!r}"
            with open(os.path.join(out_dir,
                                   f"key_averages_{label}_{tag}.txt"),
                      "w", encoding="utf-8") as handle:
                handle.write(tables[label])

    payload = {
        "mode": "profile",
        "variant": bool(args.variant),
        "tag": tag,
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
    print("profile", tag, "exported", exported["called"],
          "trace", payload["trace_bytes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
