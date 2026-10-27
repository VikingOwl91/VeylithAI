"""PERF-5G experimental wrapper: pad the w3 input-gradient output storage.

The fault buffer is the input gradient of the expert ``w3`` linear: shape
(3075, 682) bf16, 4,194,300 bytes, which ATen allocates inside ``MmBackward``.
There is no Python hook that supplies an output buffer to that node, and
``torch.mm(..., out=)`` does not participate in autograd, so the only way to
control the allocation is to replace the node with a custom autograd Function
that performs the *same two GEMMs* and writes the input gradient into a padded
storage.

Scope: experimental only, applied at runtime to ``SwiGLU.w3`` modules of the
reproducer's model. No production module is touched.

Modes
    selftest   CPU: bitwise comparison of the stock and wrapped gradients in
               fp32 and bf16, plus the padded-buffer geometry checks
    repro      run the reproducer, with ``--pad 0`` (reference) or ``--pad 1``
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.path.join(HERE, "..", "..", "debug")
sys.path.insert(0, DEBUG)

MIB = 2 << 20
SLACK_BYTES = 64 * 1024
GEOMETRY = []


def padded_buffer(rows: int, cols: int, dtype, device, slack: int = SLACK_BYTES):
    """Allocate one storage and return a (rows, cols) view with trailing slack."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    tensor_bytes = rows * cols * itemsize
    total = tensor_bytes + slack
    raw = torch.empty(total, dtype=torch.uint8, device=device)
    view = raw[:tensor_bytes].view(dtype).view(rows, cols)
    storage = view.untyped_storage()
    tensor_end = view.data_ptr() + tensor_bytes
    storage_end = storage.data_ptr() + storage.nbytes()
    page_end = ((storage_end + 4095) // 4096) * 4096
    row = {
        "shape": [rows, cols], "dtype": str(dtype),
        "data_ptr": hex(view.data_ptr()), "storage_ptr": hex(storage.data_ptr()),
        "storage_bytes": storage.nbytes(), "tensor_bytes": tensor_bytes,
        "storage_offset": view.storage_offset(),
        "stride": list(view.stride()), "contiguous": view.is_contiguous(),
        "slack_after_tensor": storage_end - tensor_end,
        "page_rounded_storage_end": hex(page_end),
        "bytes_from_tensor_end_to_page_end": page_end - tensor_end,
        "t": round(time.time(), 3),
    }
    GEOMETRY.append(row)
    if os.environ.get("V31P5G_PRINT_ALLOC") == "1":
        print("PADALLOC " + json.dumps(row, sort_keys=True), flush=True)
    return view


class PaddedW3(torch.autograd.Function):
    """Same GEMMs as ATen's linear backward, input gradient into padded storage."""

    @staticmethod
    def forward(ctx, x, weight):
        # Autocast would cast the fp32 master weight to the activation dtype; do
        # it explicitly so the same computation also runs outside autocast.
        ctx.save_for_backward(x, weight)
        weight_cast = weight if weight.dtype == x.dtype else weight.to(x.dtype)
        return F.linear(x, weight_cast)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # grad_input has x's shape: rows of grad_output, weight's input width.
        # The buffer is allocated with that exact shape so mm(out=) never resizes
        # it (a resize would silently discard the padding).
        # Under autocast the forward used a bf16 copy of the fp32 master weight,
        # so the backward must do the same; ATen reaches this through its own
        # cast node. The weight gradient is produced in the activation dtype and
        # returned in the parameter dtype, matching that cast.
        weight_cast = weight.to(grad_output.dtype)
        # ATen's linear backward flattens leading dimensions before the GEMMs;
        # linear also receives 3-D input from the dense blocks, so do the same and
        # view the result back. The view is free, the storage is the padded one.
        x2 = x.reshape(-1, x.shape[-1])
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        out = padded_buffer(g2.shape[0], weight.shape[1],
                            grad_output.dtype, grad_output.device)
        expected_ptr = out.data_ptr()
        expected_storage = out.untyped_storage().nbytes()
        shape_before = tuple(out.shape)
        grad_input = torch.mm(g2, weight_cast, out=out)
        GEOMETRY[-1]["pointer_preserved"] = grad_input.data_ptr() == expected_ptr
        GEOMETRY[-1]["storage_preserved"] = (grad_input.untyped_storage().nbytes()
                                             == expected_storage)
        GEOMETRY[-1]["resized_by_out"] = tuple(grad_input.shape) != shape_before
        grad_weight = torch.mm(g2.t(), x2)
        GEOMETRY[-1]["weight_cast_used"] = str(weight_cast.dtype)
        return grad_input.view(x.shape), grad_weight.to(weight.dtype)


def _padded_forward(self, x):
    return PaddedW3.apply(x, self.weight)


def apply_padding(model) -> int:
    """Give each ``SwiGLU.w3`` a padded-backward forward, keeping the module.

    The module has to stay an ``nn.Linear``. ``GPT._init_weights`` initializes
    only instances of ``nn.Linear`` and ``nn.Embedding``, so replacing the
    module with a wrapper skips w3's initialization *and* shifts the RNG stream
    for every parameter initialized after it. That is a model-state change, not
    a storage change, and it made PERF-5H's first arm numerically invalid.

    Patching only the bound ``forward`` keeps the class, the parameter object,
    the parameter names and the initialization path identical.
    """
    patched = 0
    for module in model.modules():
        w3 = getattr(module, "w3", None)
        if isinstance(w3, torch.nn.Linear) and "forward" not in w3.__dict__:
            w3.forward = types.MethodType(_padded_forward, w3)
            patched += 1
    return patched


def unapply_padding(model) -> int:
    """Restore the stock ``nn.Linear`` forward on every patched module."""
    removed = 0
    for module in model.modules():
        w3 = getattr(module, "w3", None)
        if isinstance(w3, torch.nn.Linear) and "forward" in w3.__dict__:
            del w3.__dict__["forward"]
            removed += 1
    return removed


def capture_first_forward(impl, store):
    """Record the first training batch, its full-precision loss and the weights.

    The reproducer prints ``{loss:.4f}`` only. Gate A wants the underlying
    value, so keep the full float and enough state to recompute it after the
    loop. The run uses ``--no_optimizer``, so the weights at the end of the loop
    are the weights at iteration 0; the check below verifies that rather than
    assuming it.
    """
    original = impl.GPT.forward

    def wrapped(self, idx, targets=None):
        out = original(self, idx, targets)
        if "model" not in store:
            store["model"] = self
            store["idx"] = idx.detach().clone()
            store["targets"] = targets.detach().clone() if targets is not None else None
            store["loss"] = float(out[1].detach())
            # Print immediately: PERF-5I faulted inside the loop, so the value
            # captured here was never persisted and Gate A could only be applied
            # at display precision.
            print("FIRSTLOSS " + json.dumps({"iter0_loss": repr(store["loss"]),
                                             "iter0_loss_4dp": f"{store['loss']:.4f}"}),
                  flush=True)
            store["params"] = {n: p.detach().clone()
                               for n, p in self.named_parameters()}
        return out

    impl.GPT.forward = wrapped
    return original


def post_check(impl, store) -> dict:
    """Recompute iteration 0 after the loop, stock and padded, at full precision.

    Runs after ``COMPLETED`` so it cannot change the fault conditions of the
    loop. The padded pass runs first, so a fault in the stock pass cannot destroy
    the padded-arm measurement.
    """
    report = {"postcheck_started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
              "postcheck_started_epoch": round(time.time(), 3)}
    model = store.get("model")
    if model is None:
        return {"error": "no captured forward"}
    report["iter0_loss_loop"] = store["loss"]
    report["iter0_loss_loop_4dp"] = f"{store['loss']:.4f}"
    report["model_state_unchanged_after_loop"] = bool(all(
        torch.equal(p.detach(), store["params"][n])
        for n, p in model.named_parameters()))
    idx, tgt = store["idx"], store["targets"]
    experts = [(n, p) for n, p in model.named_parameters() if ".mlp.experts." in n]

    def run(patched: bool):
        if patched:
            apply_padding(model)
        else:
            unapply_padding(model)
        report["w3_modules_padded_at_pass" if patched else
               "w3_modules_stock_at_pass"] = sum(
            1 for m in model.modules()
            if isinstance(getattr(m, "w3", None), torch.nn.Linear)
            and "forward" in m.w3.__dict__)
        model.zero_grad(set_to_none=True)
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if idx.is_cuda else contextlib.nullcontext())
        with autocast:
            _, loss = model(idx, tgt)
        value = float(loss.detach())
        if loss.requires_grad:
            loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in experts if p.grad is not None}
        norm = float(sum(float(g.double().pow(2).sum()) for g in grads.values()) ** 0.5)
        model.zero_grad(set_to_none=True)
        return value, grads, norm

    try:
        pad_loss, pad_grads, pad_norm = run(True)
        report["padded_loss_recomputed"] = pad_loss
        report["padded_grad_norm_experts"] = pad_norm
        report["padded_grads_captured"] = len(pad_grads)
        report["padded_loss_equals_loop_loss"] = pad_loss == store["loss"]
    except BaseException as exc:  # noqa: BLE001
        report["padded_pass_error"] = f"{type(exc).__name__}: {exc}"
        return report
    try:
        stock_loss, stock_grads, stock_norm = run(False)
        report["stock_loss_recomputed"] = stock_loss
        report["stock_grad_norm_experts"] = stock_norm
        report["stock_grads_captured"] = len(stock_grads)
        report["stock_vs_padded_loss_bitwise"] = stock_loss == pad_loss
        report["stock_vs_padded_loss_4dp_equal"] = (
            f"{stock_loss:.4f}" == f"{pad_loss:.4f}")
        report["grad_names_equal"] = sorted(stock_grads) == sorted(pad_grads)
        report["grads_bitwise_equal"] = bool(
            report["grad_names_equal"]
            and all(torch.equal(stock_grads[n], pad_grads[n]) for n in stock_grads))
        report["grad_norm_equal"] = stock_norm == pad_norm
    except BaseException as exc:  # noqa: BLE001
        report["stock_pass_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _debug_config():
    import sparse_moe_v2_train as impl

    return impl.Config(n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
                       sliding_window=128, full_attn_every=4, dropout=0.0,
                       n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
                       vocab_size=1000)


def full_model_gate():
    """Stock versus patched model: bitwise state, forward and backward on CPU.

    This is the gate that PERF-5H needed: it compares the *whole model state*
    after construction, not only the GEMM outputs, so a patching mechanism that
    changes initialization cannot pass.
    """
    import sparse_moe_v2_train as impl

    torch.manual_seed(1337)
    stock = impl.GPT(_debug_config())
    original_init = impl.GPT.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        apply_padding(self)

    impl.GPT.__init__ = patched_init
    try:
        torch.manual_seed(1337)
        pad_model = impl.GPT(_debug_config())
    finally:
        impl.GPT.__init__ = original_init

    s, p = dict(stock.named_parameters()), dict(pad_model.named_parameters())
    names_equal = list(s) == list(p)
    state_identical = names_equal and all(torch.equal(s[n], p[n]) for n in s)
    patched_modules = sum(1 for m in pad_model.modules() if "forward" in m.__dict__)
    linear_kept = all(isinstance(m.w3, torch.nn.Linear)
                      for m in pad_model.modules() if hasattr(m, "w3"))

    gen = torch.Generator().manual_seed(7)
    idx = torch.randint(0, 1000, (2, 32), generator=gen)
    tgt = torch.randint(0, 1000, (2, 32), generator=gen)
    logits_a, loss_a = stock(idx, tgt)
    loss_a.backward()
    logits_b, loss_b = pad_model(idx, tgt)
    loss_b.backward()
    pairs = list(zip(stock.parameters(), pad_model.parameters()))
    grads_bitwise = all(
        (torch.equal(a.grad, b.grad) if a.grad is not None and b.grad is not None
         else a.grad is None and b.grad is None) for a, b in pairs)
    return {
        "params": len(s), "names_equal": names_equal,
        "state_bitwise_identical": bool(state_identical),
        "logits_bitwise": bool(torch.equal(logits_a, logits_b)),
        "loss_bitwise": bool(torch.equal(loss_a, loss_b)),
        "grads_bitwise": bool(grads_bitwise),
        "loss": float(loss_a.detach()),
        "w3_forwards_patched": patched_modules,
        "all_w3_still_linear": bool(linear_kept),
    }


def post_check_gate():
    """Exercise the post-loop comparison logic on CPU before any GPU use."""
    import sparse_moe_v2_train as impl

    torch.manual_seed(11)
    model = impl.GPT(_debug_config())
    apply_padding(model)
    gen = torch.Generator().manual_seed(13)
    idx = torch.randint(0, 1000, (2, 32), generator=gen)
    tgt = torch.randint(0, 1000, (2, 32), generator=gen)
    with torch.no_grad():
        _, loss = model(idx, tgt)
    store = {"model": model, "idx": idx, "targets": tgt,
             "loss": float(loss.detach()),
             "params": {n: p.detach().clone() for n, p in model.named_parameters()}}
    res = post_check(impl, store)
    keys = ("model_state_unchanged_after_loop", "padded_loss_equals_loop_loss",
            "stock_vs_padded_loss_bitwise", "grads_bitwise_equal",
            "grad_norm_equal", "padded_grads_captured", "stock_grads_captured")
    out = {k: res.get(k) for k in keys}
    out["stock_loss_minus_padded"] = (res.get("stock_loss_recomputed", 0.0)
                                      - res.get("padded_loss_recomputed", 0.0))
    out["errors"] = [res[k] for k in ("padded_pass_error", "stock_pass_error")
                     if k in res]
    return out


def selftest() -> int:
    torch.manual_seed(0)
    report = {"cases": []}
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(37, 16, dtype=dtype)
        w = torch.randn(11, 16, dtype=dtype)
        x1 = x.clone().requires_grad_(True)
        w1 = w.clone().requires_grad_(True)
        y1 = F.linear(x1, w1)
        y1.sum().backward()
        stock_gi, stock_gw = x1.grad.clone(), w1.grad.clone()

        x2 = x.clone().requires_grad_(True)
        w2 = w.clone().requires_grad_(True)
        y2 = PaddedW3.apply(x2, w2)
        y2.sum().backward()
        report["cases"].append({
            "dtype": str(dtype),
            "forward_bitwise": bool(torch.equal(y1.detach(), y2.detach())),
            "grad_input_bitwise": bool(torch.equal(stock_gi, x2.grad)),
            "grad_weight_bitwise": bool(torch.equal(stock_gw, w2.grad)),
            "grad_input_max_abs": float((stock_gi.float() - x2.grad.float()).abs().max()),
            "grad_weight_max_abs": float((stock_gw.float() - w2.grad.float()).abs().max()),
        })
    # mixed dtype, as autocast produces: bf16 activations with an fp32 master
    # weight. The stock reference is the explicit autocast equivalent: a bf16
    # weight copy, with the weight gradient cast back to fp32.
    xb = torch.randn(23, 16, dtype=torch.bfloat16)
    wf = torch.randn(11, 16, dtype=torch.float32)
    w_bf16 = wf.detach().clone().to(torch.bfloat16).requires_grad_(True)
    xb1 = xb.clone().requires_grad_(True)
    F.linear(xb1, w_bf16).sum().backward()
    stock_gi = xb1.grad.clone()
    stock_gw = w_bf16.grad.clone().to(torch.float32)
    xb2 = xb.clone().requires_grad_(True)
    wf2 = wf.clone().requires_grad_(True)
    PaddedW3.apply(xb2, wf2).sum().backward()
    mixed_row = GEOMETRY[-1]
    report["mixed_dtype"] = {
        "grad_input_bitwise": bool(torch.equal(stock_gi, xb2.grad)),
        "grad_weight_bitwise": bool(torch.equal(stock_gw, wf2.grad)),
        "grad_input_dtype": str(xb2.grad.dtype), "grad_weight_dtype": str(wf2.grad.dtype),
        "weight_cast_used": mixed_row.get("weight_cast_used"),
        "resized_by_out": mixed_row.get("resized_by_out"),
        "pointer_preserved": mixed_row.get("pointer_preserved"),
        "storage_preserved": mixed_row.get("storage_preserved"),
        "storage_offset": mixed_row.get("storage_offset"),
        "contiguous": mixed_row.get("contiguous"),
    }

    # explicit out= semantics check: the write must land in the padded view and
    # the returned tensor must keep its pointer and storage
    a = torch.randn(37, 11)
    b = torch.randn(11, 16)
    reference = a @ b
    probe = padded_buffer(37, 16, torch.float32, "cpu")
    probe_buf_bytes = probe.untyped_storage().nbytes()
    result = torch.mm(a, b, out=probe)
    out_wrote_into_view = bool(torch.equal(probe, reference))
    out_pointer_preserved = result.data_ptr() == probe.data_ptr()
    out_storage_preserved = result.untyped_storage().nbytes() == probe_buf_bytes
    out_returned_same_object = result is probe

    # geometry of one padded buffer
    view = padded_buffer(64, 32, torch.bfloat16, "cpu")
    geo = GEOMETRY[-1]
    report["geometry"] = geo
    report["geometry_ok"] = (geo["contiguous"] and geo["storage_offset"] == 0
                             and geo["slack_after_tensor"] >= SLACK_BYTES)
    report["all_bitwise"] = all(c["forward_bitwise"] and c["grad_input_bitwise"]
                                and c["grad_weight_bitwise"] for c in report["cases"])
    report["all_bitwise"] = report["all_bitwise"] and all(
        report["mixed_dtype"][k] for k in ("grad_input_bitwise", "grad_weight_bitwise"))
    report["pointer_preserved"] = out_pointer_preserved
    report["storage_preserved"] = out_storage_preserved
    report["out_wrote_into_view"] = out_wrote_into_view
    report["out_returned_same_object"] = out_returned_same_object
    post = post_check_gate()
    report["post_check"] = post
    post_ok = (post["model_state_unchanged_after_loop"] is True
               and post["padded_loss_equals_loop_loss"] is True
               and post["stock_vs_padded_loss_bitwise"] is True
               and post["grads_bitwise_equal"] is True
               and post["grad_norm_equal"] is True
               and (post["padded_grads_captured"] or 0) > 0
               and (post["stock_grads_captured"] or 0) > 0
               and not post["errors"])
    report["post_check_ok"] = post_ok

    model = full_model_gate()
    report["full_model"] = model
    model_ok = (model["names_equal"] and model["state_bitwise_identical"]
                and model["logits_bitwise"] and model["loss_bitwise"]
                and model["grads_bitwise"] and model["all_w3_still_linear"]
                and model["w3_forwards_patched"] > 0)
    report["full_model_ok"] = model_ok
    mixed_ok = (report["mixed_dtype"]["grad_input_bitwise"]
                and report["mixed_dtype"]["grad_weight_bitwise"]
                and report["mixed_dtype"]["weight_cast_used"] == "torch.bfloat16"
                and report["mixed_dtype"]["grad_input_dtype"] == "torch.bfloat16"
                and report["mixed_dtype"]["grad_weight_dtype"] == "torch.float32"
                and report["mixed_dtype"]["resized_by_out"] is False
                and report["mixed_dtype"]["pointer_preserved"] is True
                and report["mixed_dtype"]["contiguous"] is True
                and report["mixed_dtype"]["storage_offset"] == 0)
    report["mixed_dtype_ok"] = mixed_ok
    print("SELFTEST " + json.dumps({k: report[k] for k in
                                    ("all_bitwise", "geometry_ok", "pointer_preserved",
                                     "storage_preserved", "out_wrote_into_view",
                                     "out_returned_same_object", "mixed_dtype_ok",
                                     "full_model_ok", "post_check_ok")})
          + " " + json.dumps(report["cases"]))
    print("MIXED " + json.dumps(report["mixed_dtype"], sort_keys=True))
    print("MODEL " + json.dumps(model, sort_keys=True))
    print("POST " + json.dumps(post, sort_keys=True))
    ok = (report["all_bitwise"] and report["geometry_ok"] and mixed_ok and model_ok
          and post_ok
          and report["pointer_preserved"] and report["storage_preserved"]
          and report["out_wrote_into_view"] and report["out_returned_same_object"])
    return 0 if ok else 1


def dump_payload(payload, out_path) -> None:
    """Persist the record.

    Called once when the loop ends and again at exit, so a fault during the
    post-loop check cannot destroy the loop evidence.
    """
    targets = [g for g in GEOMETRY if g["shape"] == [3075, 682]]
    payload["buffers"] = list(GEOMETRY)
    payload["buffers_total"] = len(GEOMETRY)
    payload["slack_min"] = min((g["slack_after_tensor"] for g in GEOMETRY),
                               default=None)
    payload["all_contiguous"] = all(g["contiguous"] for g in GEOMETRY)
    payload["all_offset_zero"] = all(g["storage_offset"] == 0 for g in GEOMETRY)
    payload["all_pointer_preserved"] = all(g.get("pointer_preserved") is True
                                           for g in GEOMETRY)
    payload["all_storage_preserved"] = all(g.get("storage_preserved") is True
                                           for g in GEOMETRY)
    payload["any_resized_by_out"] = any(g.get("resized_by_out") is True
                                        for g in GEOMETRY)
    payload["target_buffers"] = len(targets)
    payload["target_bytes_to_page_end_min"] = min(
        (g["bytes_from_tensor_end_to_page_end"] for g in targets), default=None)
    json.dump(payload, open(out_path, "w"), indent=2, sort_keys=True)


def repro(pad: int, out_path: str, repro_args) -> int:
    import reproducer
    import sparse_moe_v2_train as sparse_impl

    GEOMETRY.clear()
    if pad:
        # Patch after construction: the module must stay an nn.Linear so that
        # _init_weights still initializes w3 inside GPT.__init__.
        original_init = sparse_impl.GPT.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            apply_padding(self)

        sparse_impl.GPT.__init__ = patched_init
    sys.argv = [os.path.join(DEBUG, "reproducer.py")] + list(repro_args)
    exit_code = 0
    payload = {"pad": bool(pad), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime())}
    store = {}
    original_forward = capture_first_forward(sparse_impl, store)
    try:
        reproducer.main()
        # The loop is over: persist its evidence before any post-loop GPU work.
        dump_payload(payload, out_path)
        print("POSTCHECK-BEGIN", flush=True)
        try:
            payload["post_check"] = post_check(sparse_impl, store)
        except BaseException as exc:  # noqa: BLE001
            payload["post_check"] = {"error": f"{type(exc).__name__}: {exc}"}
    except BaseException as exc:  # noqa: BLE001
        payload["exception"] = {"type": type(exc).__name__,
                                "message": str(exc).splitlines()[0][:300]}
        exit_code = 1
    finally:
        sparse_impl.GPT.forward = original_forward
        dump_payload(payload, out_path)
    pc = payload.get("post_check") or {}
    print("REPRO " + json.dumps({"pad": bool(pad), "buffers": len(GEOMETRY),
                                 "slack_min": payload.get("slack_min"),
                                 "exception": payload.get("exception"),
                                 "post_check": {k: pc.get(k) for k in
                                                ("iter0_loss_loop_4dp",
                                                 "padded_loss_recomputed",
                                                 "stock_loss_recomputed",
                                                 "padded_loss_equals_loop_loss",
                                                 "stock_vs_padded_loss_bitwise",
                                                 "grads_bitwise_equal")}}), flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selftest", "repro"), required=True)
    parser.add_argument("--pad", type=int, choices=(0, 1), default=0)
    parser.add_argument("--out", default="geometry.json")
    parser.add_argument("--repro-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.mode == "selftest":
        return selftest()
    return repro(args.pad, args.out, args.repro_args)


if __name__ == "__main__":
    raise SystemExit(main())
