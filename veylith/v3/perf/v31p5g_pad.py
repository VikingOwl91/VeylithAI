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
                                     "full_model_ok")})
          + " " + json.dumps(report["cases"]))
    print("MIXED " + json.dumps(report["mixed_dtype"], sort_keys=True))
    print("MODEL " + json.dumps(model, sort_keys=True))
    ok = (report["all_bitwise"] and report["geometry_ok"] and mixed_ok and model_ok
          and report["pointer_preserved"] and report["storage_preserved"]
          and report["out_wrote_into_view"] and report["out_returned_same_object"])
    return 0 if ok else 1


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
    try:
        reproducer.main()
    except BaseException as exc:  # noqa: BLE001
        payload["exception"] = {"type": type(exc).__name__,
                                "message": str(exc).splitlines()[0][:300]}
        exit_code = 1
    finally:
        # Every allocation is recorded: Gate B has to be checked per target
        # buffer, not on a sample of twenty.
        targets = [g for g in GEOMETRY if g["shape"] == [3075, 682]]
        payload["buffers"] = GEOMETRY
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
    print("REPRO " + json.dumps({"pad": bool(pad), "buffers": len(GEOMETRY),
                                 "slack_min": payload.get("slack_min"),
                                 "exception": payload.get("exception")}), flush=True)
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
