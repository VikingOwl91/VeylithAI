#!/usr/bin/env python3
"""Bounded GPU compatibility smoke test for veylith v3.1-A.

Phases: BF16 baseline, SDPA parity, sparse MoE, full Architecture A.
Stops on the first failure. No retries, no checkpoints, no dense fallback.
Run under the GPU venv, from the repository root:

    ~/veylith-v31-gpu-env/bin/python veylith/v3/gpu_smoke.py \
        --out ~/veylith-v31-gpu-smoke-20261027

Total budget 30 minutes, 120 seconds per step. All artifacts stay outside
Git; the summary JSON is small and contains no secrets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from architecture.config import get_candidate, tiny_config
from architecture.model import Attention, GPT, MoE

DEADLINE = None
STEP_TIMEOUT = 120
STARTED = None


class StepTimeout(RuntimeError):
    pass


def _alarm(signum, frame):
    raise StepTimeout(f"step exceeded {STEP_TIMEOUT}s")


def _guard():
    signal.setitimer(signal.ITIMER_REAL, STEP_TIMEOUT)


def _clear():
    signal.setitimer(signal.ITIMER_REAL, 0)


def _check_deadline(phase: str) -> None:
    if DEADLINE is not None and time.time() > DEADLINE:
        raise TimeoutError(f"30-minute budget exhausted before {phase}")


def _write(path: str, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 4),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 4),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 4),
        "free_gb": round(free / 1e9, 4),
        "total_gb": round(total / 1e9, 4),
    }


def _finite(tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _sync() -> None:
    torch.cuda.synchronize()


def _timed(fn):
    _sync()
    start = time.perf_counter()
    result = fn()
    _sync()
    return result, time.perf_counter() - start


def _router_stats(router, n_experts: int) -> dict:
    layers = []
    for top_idx in router:
        counts = torch.bincount(top_idx.reshape(-1).cpu(), minlength=n_experts).float()
        probs = counts / counts.sum()
        entropy = float(-(probs * probs.clamp(min=1e-12).log()).sum())
        layers.append({
            "entropy_norm": entropy / math.log(n_experts),
            "share_pct": [round(float(value) * 100, 2) for value in probs],
            "unused_experts": int((probs == 0).sum()),
        })
    return {"layers": layers}


def phase_baseline() -> dict:
    _check_deadline("baseline")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    results = {}
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    _guard()
    try:
        (out, fwd), fwd_s = _timed(lambda: (a @ b, None))
        loss = out.float().square().mean()
        _sync()
        start = time.perf_counter()
        loss.backward()
        _sync()
        bwd_s = time.perf_counter() - start
    finally:
        _clear()
    assert _finite(out) and _finite(a.grad)
    results["matmul_bf16"] = {"forward_s": round(fwd_s, 6),
                              "forward_backward_s": round(bwd_s, 6),
                              "grad_finite": True}
    # minimal causal attention
    q = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(64, 64, device="cuda", dtype=torch.bool).tril()
    _sync()
    start = time.perf_counter()
    att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    _sync()
    att_s = time.perf_counter() - start
    assert _finite(att)
    results["causal_sdpa"] = {"seconds": round(att_s, 6),
                              "shape": list(att.shape), "finite": True}
    results["memory"] = _mem()
    results["gpu_name"] = torch.cuda.get_device_name(0)
    results["torch"] = torch.__version__
    results["hip"] = torch.version.hip
    results["python"] = sys.version.split()[0]
    results["platform"] = platform.platform()
    return results


def _attention_pair(layer_idx: int, dtype, seq_len: int):
    base = get_candidate("A")
    base.dropout = 0.0
    base.attention_impl = "manual"
    manual = Attention(base, layer_idx).to("cuda").to(dtype)
    cfg_sdpa = get_candidate("A")
    cfg_sdpa.dropout = 0.0
    cfg_sdpa.attention_impl = "sdpa"
    sdpa = Attention(cfg_sdpa, layer_idx).to("cuda").to(dtype)
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, seq_len, base.n_embd, device="cuda", dtype=dtype)
    return manual, sdpa, x


def _parity(manual, sdpa, x, atol, rtol):
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    manual.zero_grad(set_to_none=True)
    sdpa.zero_grad(set_to_none=True)
    y1 = manual(x1)
    y2 = sdpa(x2)
    assert _finite(y1) and _finite(y2)
    max_abs = float((y1.float() - y2.float()).abs().max())
    scale = float(y1.float().abs().max().clamp(min=1e-6))
    y1.float().pow(2).mean().backward()
    y2.float().pow(2).mean().backward()
    grad_abs = float((x1.grad.float() - x2.grad.float()).abs().max())
    param_diff = max(float((p1.detach().float() - p2.detach().float()).abs().max())
                     for p1, p2 in zip(manual.parameters(), sdpa.parameters()))
    return {"max_abs_diff": round(max_abs, 6),
            "max_rel_diff": round(max_abs / scale, 6),
            "input_grad_max_abs_diff": round(grad_abs, 6),
            "param_copy_diff": round(param_diff, 6),
            "tolerance": {"atol": atol, "rtol": rtol}}


def phase_sdpa() -> dict:
    _check_deadline("sdpa")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1)
    results = {}
    for dtype_name, dtype, atol, rtol in (
            ("fp32", torch.float32, 1e-4, 1e-4),
            ("bf16", torch.bfloat16, 2e-2, 2e-2)):
        for seq_len in (128, 512):
            for layer_idx, mask in ((3, "full"), (0, "sliding")):
                manual, sdpa, x = _attention_pair(layer_idx, dtype, seq_len)
                _guard()
                try:
                    entry = _parity(manual, sdpa, x, atol, rtol)
                finally:
                    _clear()
                key = f"{dtype_name}_T{seq_len}_{mask}"
                results[key] = entry
                if entry["max_abs_diff"] > atol or entry["input_grad_max_abs_diff"] > atol:
                    raise AssertionError(f"SDPA parity failed for {key}: {entry}")
    results["memory"] = _mem()
    return results


def _moe_case(name: str, cfg, seq_len: int, batch: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(2)
    moe = MoE(cfg).to("cuda").to(torch.bfloat16)
    x = torch.randn(batch, seq_len, cfg.n_embd, device="cuda",
                    dtype=torch.bfloat16, requires_grad=True)
    moe.zero_grad(set_to_none=True)
    _guard()
    try:
        (out, aux, top_idx), fwd_s = _timed(lambda: moe(x))
        assert _finite(out) and _finite(aux)
        (loss, _), bwd_s = _timed(lambda: (out.float().pow(2).mean() + 0.01 * aux, None))
        loss.backward()
    finally:
        _clear()
    assert _finite(x.grad)
    grads = [p.grad for p in moe.parameters() if p.grad is not None]
    assert grads and all(_finite(g) for g in grads)
    assert top_idx.shape == (batch * seq_len, cfg.n_experts_per_tok)
    for row in top_idx[: min(16, top_idx.shape[0])]:
        assert len(set(row.tolist())) == cfg.n_experts_per_tok
    counts = torch.bincount(top_idx.reshape(-1).cpu(), minlength=cfg.n_experts).float()
    probs = counts / counts.sum()
    return {"forward_s": round(fwd_s, 6), "backward_s": round(bwd_s, 6),
            "aux": round(float(aux.detach()), 6),
            "topk_shape": list(top_idx.shape),
            "router_entropy_norm": round(
                float(-(probs * probs.clamp(min=1e-12).log()).sum()) / math.log(cfg.n_experts), 4),
            "memory": _mem()}


def phase_moe() -> dict:
    _check_deadline("moe")
    results = {}
    minimal = tiny_config(n_embd=64, n_head=4, n_kv_head=2, n_experts=4,
                          n_experts_per_tok=2, expert_ffn_dim=128,
                          shared_expert=True, shared_ffn_dim=128,
                          first_k_dense=0, block_size=64, dropout=0.0)
    results["minimal"] = _moe_case("minimal", minimal, 32, 1)
    wide = tiny_config(n_embd=512, n_head=8, n_kv_head=2, n_experts=8,
                       n_experts_per_tok=2, expert_ffn_dim=960,
                       shared_expert=False, shared_ffn_dim=0,
                       first_k_dense=0, block_size=512, dropout=0.1)
    results["architecture_a_width"] = _moe_case("a-width", wide, 64, 1)
    return results


def _full_step(model, optimizer, x, y, cfg, precision="bf16"):
    optimizer.zero_grad(set_to_none=True)
    _sync()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, metrics, router = model(x, targets=y, return_metrics=True,
                                           return_router=True)
    _sync()
    fwd = time.perf_counter() - start
    assert _finite(logits) and _finite(metrics["composite_loss"])
    loss = metrics["composite_loss"]
    _sync()
    start = time.perf_counter()
    loss.backward()
    _sync()
    bwd = time.perf_counter() - start
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(_finite(g) for g in grads)
    _sync()
    start = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    _sync()
    opt = time.perf_counter() - start
    return {"forward_s": round(fwd, 4), "backward_s": round(bwd, 4),
            "optimizer_s": round(opt, 4), "step_s": round(fwd + bwd + opt, 4),
            "main_ce": round(float(metrics["main_ce"].detach()), 4),
            "aux": round(float(metrics["aux_loss"].detach()), 4),
            "router": _router_stats(router, model.cfg.n_experts)}


def phase_full() -> dict:
    _check_deadline("full")
    model_cfg = get_candidate("A")
    model_cfg.attention_impl = "sdpa"
    model_cfg.moe_impl = "sparse"
    torch.manual_seed(3)
    model = GPT(model_cfg).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    results = {"config": model_cfg.to_dict(), "microbatches": {}}
    seq = 512
    for microbatch in (1, 2, 4, 8):
        _check_deadline(f"full mb={microbatch}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(100 + microbatch)
        steps = []
        for step in range(5):
            _check_deadline(f"full mb={microbatch} step {step}")
            ids = torch.randint(0, model_cfg.vocab_size, (microbatch, seq),
                                device="cuda")
            x, y = ids[:, :-1], ids[:, 1:]
            _guard()
            try:
                entry = _full_step(model, optimizer, x, y, model_cfg)
            finally:
                _clear()
            steps.append(entry)
            if entry["step_s"] > STEP_TIMEOUT:
                raise StepTimeout(f"mb={microbatch} step {step} took {entry['step_s']}s")
        results["microbatches"][str(microbatch)] = {
            "steps": steps,
            "median_step_s": round(sorted(s["step_s"] for s in steps)[2], 4),
            "tokens_per_s": round(microbatch * (seq - 1) / max(1e-9, sorted(
                s["step_s"] for s in steps)[2]), 1),
            "memory": _mem(),
        }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    global DEADLINE, STARTED
    STARTED = time.time()
    DEADLINE = STARTED + args.budget_seconds
    signal.signal(signal.SIGALRM, _alarm)
    summary = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "phases": {}, "status": "running"}
    summary_path = os.path.join(args.out, "summary.json")
    _write(summary_path, summary)
    phase = "preflight"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is not available")
        summary["device"] = {
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
            "torch": torch.__version__, "hip": torch.version.hip,
            "python": sys.version.split()[0],
            "device_count": torch.cuda.device_count(),
        }
        for phase, fn in (("baseline", phase_baseline),
                          ("sdpa", phase_sdpa),
                          ("moe", phase_moe),
                          ("full", phase_full)):
            _check_deadline(phase)
            summary["phases"][phase] = fn()
            summary["status"] = f"ok:{phase}"
            _write(summary_path, summary)
            print(f"phase {phase}: ok")
        summary["status"] = "ok"
        summary["elapsed_s"] = round(time.time() - STARTED, 1)
        _write(summary_path, summary)
        return 0
    except BaseException as exc:
        _clear()
        try:
            memory = _mem() if torch.cuda.is_available() else None
        except Exception:
            memory = {"error": "memory query failed after the stop"}
        error = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_s": round(time.time() - STARTED, 1),
            "last_success": summary.get("status"),
            "memory": memory,
        }
        summary["status"] = "error"
        summary["error"] = error
        _write(summary_path, summary)
        _write(os.path.join(args.out, "error.json"), error)
        print(f"STOP at {phase}: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
