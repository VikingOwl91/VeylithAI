"""PERF-5L CPU isolation probe: does mode 1 pad only the product gradient?

Mode 1 (``PaddedW3``) replaces the input-gradient GEMM of each ``SwiGLU.w3``
linear. For that linear the input *is* the SwiGLU product, so its input gradient
is dL/d(product). This probe checks, tensor by tensor, that the forward product
stays a stock allocation and that the gradient handed back to the product is a
harness-padded buffer, at the moment each tensor exists rather than by address
after the fact.

Read-only: it imports the harness and records what autograd does.
"""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "veylith/debug")
sys.path.insert(0, "veylith/v3/perf")
import sparse_moe_v2_train as impl  # noqa: E402
import v31p5g_pad as harness  # noqa: E402


def cfg():
    return impl.Config(n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
                       sliding_window=128, full_attn_every=4, dropout=0.0,
                       n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
                       vocab_size=1000)


def padded_now():
    return {r["data_ptr"]: r for r in harness.GEOMETRY}


def main() -> int:
    torch.manual_seed(1)
    model = impl.GPT(cfg())
    harness.GEOMETRY.clear()
    harness.apply_mode(model, 1)

    seen = {"w3_calls": 0, "product_stock": 0, "product_padded": 0,
            "product_storage_exact": 0, "grad_padded": 0, "grad_stock": 0,
            "grad_kind_w3_grad": 0, "grad_slack_ok": 0, "grad_offset_zero": 0,
            "grad_contiguous": 0, "grad_product_kind_present": 0}
    product_tails, grad_tails, shapes = [], [], set()

    def on_product_grad(g):
        rec = padded_now().get(hex(g.data_ptr()))
        seen["grad_padded" if rec else "grad_stock"] += 1
        if rec:
            seen["grad_kind_w3_grad"] += int(rec.get("kind") == "w3_grad")
            seen["grad_product_kind_present"] += int(rec.get("kind") == "swiglu_product")
            seen["grad_slack_ok"] += int(rec["slack_after_tensor"] >= 65536)
            seen["grad_offset_zero"] += int(rec["storage_offset"] == 0)
            seen["grad_contiguous"] += int(rec["contiguous"])
            end = g.data_ptr() + g.numel() * g.element_size()
            grad_tails.append(((end + (2 << 20) - 1) // (2 << 20)) * (2 << 20) - end)

    def pre_hook(module, inputs):
        t = inputs[0]
        seen["w3_calls"] += 1
        shapes.add(tuple(t.shape))
        rec = padded_now().get(hex(t.data_ptr()))
        seen["product_padded" if rec else "product_stock"] += 1
        seen["product_storage_exact"] += int(
            t.untyped_storage().nbytes() == t.numel() * t.element_size())
        end = t.data_ptr() + t.numel() * t.element_size()
        product_tails.append(((end + (2 << 20) - 1) // (2 << 20)) * (2 << 20) - end)
        t.register_hook(on_product_grad)

    for m in model.modules():
        if isinstance(getattr(m, "w3", None), torch.nn.Linear):
            m.w3.register_forward_pre_hook(pre_hook)

    gen = torch.Generator().manual_seed(5)
    idx = torch.randint(0, 1000, (2, 32), generator=gen)
    tgt = torch.randint(0, 1000, (2, 32), generator=gen)
    _, loss = model(idx, tgt)
    loss.backward()

    kinds = sorted({r.get("kind") for r in harness.GEOMETRY})
    report = {
        "mode": 1, "padded_buffers": len(harness.GEOMETRY), "kinds_recorded": kinds,
        "modules_with_patched_forward": sum(1 for m in model.modules()
                                            if "forward" in m.__dict__),
        "shapes_seen": sorted(list(s) for s in shapes)[:4],
        **seen,
        "product_tail_histogram": sorted({t: product_tails.count(t)
                                          for t in product_tails}.items())[:6],
        "grad_tail_histogram": sorted({t: grad_tails.count(t)
                                       for t in grad_tails}.items())[:6],
        "no_swiglu_product_records": "swiglu_product" not in kinds,
        "any_resize": any(r.get("resized_by_out") for r in harness.GEOMETRY),
        "all_grad_pointer_preserved": all(r.get("pointer_preserved")
                                          for r in harness.GEOMETRY),
    }
    report["isolation_ok"] = bool(
        report["w3_calls"] > 0
        and report["product_padded"] == 0
        and report["product_stock"] == report["w3_calls"]
        and report["product_storage_exact"] == report["w3_calls"]
        and report["grad_padded"] == report["w3_calls"]
        and report["grad_stock"] == 0
        and report["grad_kind_w3_grad"] == report["w3_calls"]
        and report["grad_product_kind_present"] == 0
        and report["grad_slack_ok"] == report["w3_calls"]
        and report["grad_offset_zero"] == report["w3_calls"]
        and report["grad_contiguous"] == report["w3_calls"]
        and report["no_swiglu_product_records"]
        and not report["any_resize"]
        and report["all_grad_pointer_preserved"])
    print("ISOLATION " + json.dumps(report, sort_keys=True))
    return 0 if report["isolation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
