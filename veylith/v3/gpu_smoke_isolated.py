#!/usr/bin/env python3
"""Bounded v3.1-I compatibility smoke for the isolated ROCm environment.

No training, no optimizer steps, no sparse-MoE dispatch, no persistent data.
Run with the isolated launcher from the repository root:

    ~/veylith-v31-isolated/bin/veylith-python \
        veylith/v3/gpu_smoke_isolated.py \
        --out ~/veylith-v31i-gpu-smoke-20261027

The total GPU budget is 300 s and the allocator peak target is 6 GiB.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import signal
import site
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from architecture.config import ModelConfig, get_candidate
from architecture.model import Attention, GPT, MoE

TOTAL_BUDGET = 300
PHASE_TIMEOUT = 60
MEMORY_LIMIT_GB = 6.0

STARTED = None
DEADLINE = None
CURRENT_PHASE = "startup"


class SmokeStop(RuntimeError):
    pass


def _alarm(signum, frame):
    raise SmokeStop(f"phase {CURRENT_PHASE} exceeded {PHASE_TIMEOUT}s")


def _check_deadline() -> None:
    if DEADLINE is not None and time.monotonic() > DEADLINE:
        raise SmokeStop("total 300 s GPU budget exhausted")


@contextmanager
def phase(name: str):
    global CURRENT_PHASE
    CURRENT_PHASE = name
    _check_deadline()
    signal.setitimer(signal.ITIMER_REAL, PHASE_TIMEOUT)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _sync() -> None:
    torch.cuda.synchronize()


def _mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    if peak_allocated > MEMORY_LIMIT_GB:
        raise SmokeStop(
            f"peak allocated {peak_allocated:.3f} GiB exceeds "
            f"{MEMORY_LIMIT_GB:.1f} GiB")
    return {
        "allocated_gb": round(allocated, 4),
        "reserved_gb": round(reserved, 4),
        "peak_allocated_gb": round(peak_allocated, 4),
        "peak_reserved_gb": round(peak_reserved, 4),
        "free_gb": round(free / 1e9, 4),
        "total_gb": round(total / 1e9, 4),
    }


def _finite(tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: str, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _maps() -> dict:
    paths = set()
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(None, 5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    paths.add(fields[5])
    except OSError:
        pass
    relevant = sorted(path for path in paths if any(
        token in path.lower() for token in
        ("/torch/lib/", "/opt/rocm", "rocm", "hip", "amd", "miopen", "rccl", "hsa")))
    opt_rocm = sorted(path for path in paths if path.startswith("/opt/rocm"))
    if opt_rocm:
        raise SmokeStop(f"unexpected system ROCm libraries: {opt_rocm[:5]}")
    return {
        "relevant": relevant,
        "relevant_count": len(relevant),
        "opt_rocm": opt_rocm,
        "opt_rocm_count": len(opt_rocm),
    }


def _diagnose_null(out_dir: str) -> dict:
    """Capture the import diagnostic in a GPU-hidden child process."""
    strace_path = os.path.join(out_dir, "import-null.strace")
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = ""
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        "strace", "-f", "-e", "trace=write", "-o", strace_path,
        sys.executable, "-c", "import torch; print(torch.__version__)",
    ]
    try:
        result = subprocess.run(
            command, env=env, text=True, capture_output=True,
            timeout=30, check=False)
        strace_text = ""
        try:
            with open(strace_path, encoding="utf-8") as handle:
                strace_text = handle.read()
        except OSError:
            pass
        null_writes = strace_text.count('write(2, "(null): No such file')
        return {
            "returncode": result.returncode,
            "stderr_has_null": "(null): No such file or directory" in result.stderr,
            "stderr_lines": result.stderr.strip().splitlines()[:3],
            "strace_null_writes": null_writes,
            "strace_path": strace_path,
            "torch_version": result.stdout.strip(),
        }
    except FileNotFoundError:
        return {"strace_available": False}
    except subprocess.TimeoutExpired:
        return {"strace_available": True, "timeout": True}


def phase_device() -> dict:
    if not torch.cuda.is_available():
        raise SmokeStop("torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    torch.cuda.init()
    _sync()
    props = torch.cuda.get_device_properties(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    arch = getattr(props, "gcnArchName", None)
    arch_list = []
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except Exception:
        pass
    if capability != (12, 0):
        raise SmokeStop(f"unexpected device capability {capability}")
    if arch and not str(arch).startswith("gfx1201"):
        raise SmokeStop(f"unexpected gcnArchName {arch}")
    return {
        "name": torch.cuda.get_device_name(0),
        "device_count": count,
        "capability": list(capability),
        "gcn_arch_name": arch,
        "arch_list": arch_list,
        "total_memory_gb": round(props.total_memory / 1e9, 3),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch_file": torch.__file__,
        "sys_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "user_site": bool(site.ENABLE_USER_SITE),
        "memory": _mem(),
    }


def phase_bf16_matmul() -> dict:
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    _sync()
    start = time.perf_counter()
    out = a @ b
    _sync()
    fwd = time.perf_counter() - start
    loss = out.float().square().mean()
    start = time.perf_counter()
    loss.backward()
    _sync()
    bwd = time.perf_counter() - start
    if not (_finite(out) and _finite(a.grad)):
        raise SmokeStop("non-finite BF16 matmul output or gradient")
    result = {
        "shape": list(out.shape),
        "forward_s": round(fwd, 6),
        "backward_s": round(bwd, 6),
        "finite": True,
        "grad_abs_max": round(float(a.grad.abs().max()), 6),
        "memory": _mem(),
    }
    del a, b, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def phase_bf16_mlp() -> dict:
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 256),
    ).to("cuda").to(torch.bfloat16)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    _sync()
    start = time.perf_counter()
    y = net(x)
    loss = y.float().square().mean()
    _sync()
    fwd = time.perf_counter() - start
    start = time.perf_counter()
    loss.backward()
    _sync()
    bwd = time.perf_counter() - start
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if not (_finite(y) and _finite(x.grad) and grads and
            all(_finite(g) for g in grads)):
        raise SmokeStop("non-finite BF16 MLP output or gradient")
    result = {
        "shape": list(y.shape),
        "forward_s": round(fwd, 6),
        "backward_s": round(bwd, 6),
        "finite": True,
        "param_grad_count": len(grads),
        "memory": _mem(),
    }
    del net, x, y, loss, grads
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _attention_case(layer_idx: int, name: str, perturb_last: bool) -> dict:
    cfg = get_candidate("A")
    cfg.attention_impl = "sdpa"
    torch.manual_seed(0)
    module = Attention(cfg, layer_idx).to("cuda").to(torch.bfloat16)
    module.eval()
    x = torch.randn(1, 512, cfg.n_embd, device="cuda",
                    dtype=torch.bfloat16, requires_grad=True)
    _sync()
    start = time.perf_counter()
    out = module(x)
    _sync()
    fwd = time.perf_counter() - start
    loss = out.float().square().mean()
    start = time.perf_counter()
    loss.backward()
    _sync()
    bwd = time.perf_counter() - start
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not (_finite(out) and _finite(x.grad) and grads and
            all(_finite(g) for g in grads)):
        raise SmokeStop(f"non-finite {name} output or gradient")

    with torch.no_grad():
        changed = x.detach().clone()
        if perturb_last:
            changed[:, -1, :] += 0.1
            unchanged_slice = slice(0, 511)
            changed_slice = slice(511, 512)
        else:
            changed[:, 0, :] += 0.1
            unchanged_slice = slice(cfg.sliding_window, 512)
            changed_slice = slice(0, cfg.sliding_window)
        alt = module(changed)
        unchanged_diff = float(
            (alt[:, unchanged_slice, :].float() -
             out.detach()[:, unchanged_slice, :].float()).abs().max())
        changed_diff = float(
            (alt[:, changed_slice, :].float() -
             out.detach()[:, changed_slice, :].float()).abs().max())
    if unchanged_diff > 2e-2:
        raise SmokeStop(
            f"{name} mask sensitivity failed: unchanged diff {unchanged_diff}")
    result = {
        "layer": layer_idx,
        "full_attention": module.full,
        "sliding_window": cfg.sliding_window,
        "forward_s": round(fwd, 6),
        "backward_s": round(bwd, 6),
        "finite": True,
        "mask_unchanged_max_abs_diff": round(unchanged_diff, 6),
        "mask_changed_max_abs_diff": round(changed_diff, 6),
        "memory": _mem(),
    }
    del module, x, out, loss, grads
    gc.collect()
    torch.cuda.empty_cache()
    return result


def phase_dense_moe() -> dict:
    cfg = get_candidate("A")
    cfg.moe_impl = "dense"
    torch.manual_seed(0)
    moe = MoE(cfg).to("cuda").to(torch.bfloat16)
    moe.train()
    x = torch.randn(1, 64, cfg.n_embd, device="cuda",
                    dtype=torch.bfloat16, requires_grad=True)
    _sync()
    start = time.perf_counter()
    out, aux, top_idx = moe(x)
    _sync()
    fwd = time.perf_counter() - start
    loss = out.float().square().mean() + 0.01 * aux
    start = time.perf_counter()
    loss.backward()
    _sync()
    bwd = time.perf_counter() - start
    grads = [p.grad for p in moe.parameters() if p.grad is not None]
    if not (_finite(out) and _finite(aux) and _finite(x.grad) and grads and
            all(_finite(g) for g in grads)):
        raise SmokeStop("non-finite Dense-Combined MoE output or gradient")
    if top_idx.shape != (64, cfg.n_experts_per_tok):
        raise SmokeStop(f"unexpected router shape {tuple(top_idx.shape)}")
    counts = torch.bincount(top_idx.reshape(-1).cpu(),
                            minlength=cfg.n_experts).float()
    probs = counts / counts.sum()
    entropy = float(-(probs * probs.clamp(min=1e-12).log()).sum())
    result = {
        "moe_impl": cfg.moe_impl,
        "experts": cfg.n_experts,
        "top_k": cfg.n_experts_per_tok,
        "forward_s": round(fwd, 6),
        "backward_s": round(bwd, 6),
        "finite": True,
        "aux": round(float(aux.detach()), 6),
        "router_entropy_norm": round(entropy / math.log(cfg.n_experts), 4),
        "router_shape": list(top_idx.shape),
        "memory": _mem(),
    }
    del moe, x, out, aux, loss, top_idx, grads
    gc.collect()
    torch.cuda.empty_cache()
    return result


def phase_checkpoint(path: str) -> dict:
    before = _sha256(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("kind") != "model":
        raise SmokeStop(f"unexpected checkpoint kind {artifact.get('kind')}")
    cfg_dict = dict(artifact["model_config"])
    original_moe = cfg_dict.get("moe_impl")
    cfg_dict["moe_impl"] = "dense"
    cfg_dict["attention_impl"] = "sdpa"
    cfg = ModelConfig(**cfg_dict)
    torch.manual_seed(0)
    model = GPT(cfg)
    model.load_state_dict(artifact["model"], strict=True)
    model.eval()
    model.to("cuda")
    ids = torch.randint(0, cfg.vocab_size, (1, 64), device="cuda")
    _sync()
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=torch.bfloat16):
        logits, _ = model(ids)
    _sync()
    fwd = time.perf_counter() - start
    if not _finite(logits):
        raise SmokeStop("non-finite checkpoint logits")
    after = _sha256(path)
    if before != after:
        raise SmokeStop("checkpoint changed during inference")
    result = {
        "checkpoint": path,
        "sha256_before": before,
        "sha256_after": after,
        "unchanged": True,
        "artifact_step": artifact.get("step"),
        "selection": artifact.get("selection"),
        "original_moe_impl": original_moe,
        "effective_moe_impl": cfg.moe_impl,
        "effective_attention_impl": cfg.attention_impl,
        "logits_shape": list(logits.shape),
        "forward_s": round(fwd, 6),
        "finite": True,
        "memory": _mem(),
    }
    del model, artifact, ids, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None) -> int:
    global STARTED, DEADLINE, PHASE_TIMEOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser(
            "~/veylith-v31-d-20261027/ckpt_best.pt"))
    parser.add_argument("--budget-seconds", type=int, default=TOTAL_BUDGET)
    parser.add_argument("--phase-seconds", type=int, default=PHASE_TIMEOUT)
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    STARTED = time.monotonic()
    DEADLINE = STARTED + args.budget_seconds
    PHASE_TIMEOUT = args.phase_seconds
    signal.signal(signal.SIGALRM, _alarm)
    summary_path = os.path.join(args.out, "summary.json")
    summary = {
        "schema_version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "phases": {},
    }
    _write(summary_path, summary)
    phase_name = "startup"

    try:
        if "veylith-v31-isolated" not in sys.prefix:
            raise SmokeStop(f"not running in the isolated prefix: {sys.prefix}")
        if site.ENABLE_USER_SITE:
            raise SmokeStop("user site packages are enabled")
        if not os.path.realpath(torch.__file__).startswith(
                os.path.realpath(sys.prefix)):
            raise SmokeStop(f"torch imported outside the venv: {torch.__file__}")

        phase_name = "import_diagnostic"
        with phase(phase_name):
            summary["phases"][phase_name] = _diagnose_null(args.out)
        summary["status"] = "ok:import_diagnostic"
        _write(summary_path, summary)
        print("phase import_diagnostic: ok")

        phase_name = "device"
        with phase(phase_name):
            torch.cuda.reset_peak_memory_stats()
            summary["phases"][phase_name] = phase_device()
        summary["status"] = "ok:device"
        _write(summary_path, summary)
        print("phase device: ok")

        for phase_name, fn in (
                ("bf16_matmul", phase_bf16_matmul),
                ("bf16_mlp", phase_bf16_mlp),
                ("sdpa_causal",
                 lambda: _attention_case(3, "sdpa_causal", True)),
                ("sdpa_sliding",
                 lambda: _attention_case(0, "sdpa_sliding", False)),
                ("dense_moe", phase_dense_moe),
                ("checkpoint", lambda: phase_checkpoint(args.checkpoint)),
                ("libraries", _maps),
        ):
            with phase(phase_name):
                summary["phases"][phase_name] = fn()
            summary["status"] = f"ok:{phase_name}"
            _write(summary_path, summary)
            print(f"phase {phase_name}: ok")

        summary["status"] = "ok"
        summary["elapsed_s"] = round(time.monotonic() - STARTED, 3)
        summary["memory"] = _mem()
        _write(summary_path, summary)
        print(f"smoke ok in {summary['elapsed_s']}s")
        return 0
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        error = {
            "phase": phase_name,
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-5000:],
            "elapsed_s": round(time.monotonic() - STARTED, 3),
            "last_success": summary.get("status"),
        }
        summary["status"] = "error"
        summary["error"] = error
        _write(summary_path, summary)
        _write(os.path.join(args.out, "error.json"), error)
        print(f"STOP at {phase_name}: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
