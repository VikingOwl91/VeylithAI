"""CPU isolation probe for the front-padded product arms (modes 3 and 4).

Checks on the real debug model that:
  * every tensor the w3 linears receive is a front-padded view: offset equal to
    the padding (65,536 or 2,097,152 bytes), zero trailing bytes, ending at its
    backing storage end,
  * those products are bitwise identical to the stock products of the same seed
    and input,
  * the product gradient stays a stock allocation (no w3_grad records exist),
  * no w3 module is patched, only the SwiGLU forward.

Read-only: it imports the harness and records what autograd does.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

sys.path.insert(0, "veylith/debug")
sys.path.insert(0, "veylith/v3/perf")
import sparse_moe_v2_train as impl  # noqa: E402
import v31p5g_pad as harness  # noqa: E402

SLACK = 65536
FRONT_2MIB = 2 << 20


def cfg():
    return impl.Config(n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
                       sliding_window=128, full_attn_every=4, dropout=0.0,
                       n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
                       vocab_size=1000)


def batch():
    gen = torch.Generator().manual_seed(5)
    idx = torch.randint(0, 1000, (2, 32), generator=gen)
    tgt = torch.randint(0, 1000, (2, 32), generator=gen)
    return idx, tgt


def collect(model, backward: bool):
    """Products seen by each w3, and the gradient pointers flowing into them."""
    products, grads, hooks = [], [], []

    def pre_hook(module, inputs):
        t = inputs[0]
        snap = {r["data_ptr"]: r for r in harness.GEOMETRY}
        rec = snap.get(hex(t.data_ptr()))
        products.append({
            "ptr": hex(t.data_ptr()),
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "storage_bytes": t.untyped_storage().nbytes(),
            "tensor_bytes": t.numel() * t.element_size(),
            "offset_bytes": t.data_ptr() - t.untyped_storage().data_ptr(),
            "ends_at_storage_end": (t.data_ptr() + t.numel() * t.element_size()
                                    == t.untyped_storage().data_ptr()
                                    + t.untyped_storage().nbytes()),
            "kind": rec.get("kind") if rec else None,
            "clone": t.detach().clone(),
        })
        if backward:
            t.register_hook(lambda g: grads.append({
                "ptr": hex(g.data_ptr()),
                "padded": hex(g.data_ptr()) in {r["data_ptr"]
                                                for r in harness.GEOMETRY}}))

    for m in model.modules():
        if isinstance(getattr(m, "w3", None), torch.nn.Linear):
            hooks.append(m.w3.register_forward_pre_hook(pre_hook))
    idx, tgt = batch()
    _, loss = model(idx, tgt)
    if backward:
        loss.backward()
    for h in hooks:
        h.remove()
    return products, grads, float(loss.detach())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, choices=(3, 4), default=3)
    ap.add_argument("--slack", type=int, default=SLACK)
    args = ap.parse_args()
    mode, slack = args.mode, args.slack
    torch.manual_seed(1)
    stock = impl.GPT(cfg())
    harness.GEOMETRY.clear()
    stock_products, _, stock_loss = collect(stock, backward=False)

    torch.manual_seed(1)
    front = impl.GPT(cfg())
    harness.GEOMETRY.clear()
    harness.apply_mode(front, mode)
    front_products, front_grads, front_loss = collect(front, backward=True)

    same_count = len(stock_products) == len(front_products)
    products_bitwise = bool(same_count and all(
        torch.equal(a["clone"], b["clone"])
        for a, b in zip(stock_products, front_products)))
    report = {
        "mode": mode, "expected_offset": slack,
        "stock_products": len(stock_products),
        "front_products": len(front_products),
        "products_bitwise_vs_stock": products_bitwise,
        "loss_bitwise_vs_stock": stock_loss == front_loss,
        "padded_records": len(harness.GEOMETRY),
        "kinds_recorded": sorted({r.get("kind") for r in harness.GEOMETRY}),
        "all_offsets_match": all(r["offset_bytes"] == slack
                                 for r in harness.GEOMETRY),
        "all_end_at_storage_end": all(r.get("ends_at_storage_end") is True
                                      for r in harness.GEOMETRY),
        "all_zero_trailing_bytes": all(r["slack_after_tensor"] == 0
                                       for r in harness.GEOMETRY),
        "all_contiguous": all(r["contiguous"] for r in harness.GEOMETRY),
        "all_pointer_preserved": all(r.get("pointer_preserved")
                                     for r in harness.GEOMETRY),
        "any_resize": any(r.get("resized_by_out") for r in harness.GEOMETRY),
        "products_seen_as_front_padded": sum(
            1 for p in front_products if str(p["kind"]).startswith("swiglu_product_front")),
        "gradients_captured": len(front_grads),
        "gradients_padded": sum(1 for g in front_grads if g["padded"]),
        "w3_modules_patched": sum(1 for m in front.modules()
                                  if isinstance(getattr(m, "w3", None), torch.nn.Linear)
                                  and "forward" in m.w3.__dict__),
        "swiglu_modules_patched": sum(1 for m in front.modules()
                                      if hasattr(m, "w1") and "forward" in m.__dict__),
    }
    report["isolation_ok"] = bool(
        report["products_bitwise_vs_stock"] and report["loss_bitwise_vs_stock"]
        and report["front_products"] == report["stock_products"] > 0
        and len(report["kinds_recorded"]) == 1
        and str(report["kinds_recorded"][0]).startswith("swiglu_product_front")
        and report["products_seen_as_front_padded"] == report["front_products"]
        and report["all_offsets_match"] and report["all_end_at_storage_end"]
        and report["all_zero_trailing_bytes"] and report["all_contiguous"]
        and report["all_pointer_preserved"] and not report["any_resize"]
        and report["gradients_padded"] == 0
        and report["w3_modules_patched"] == 0
        and report["swiglu_modules_patched"] > 0)
    print("ISOLATION " + json.dumps(report, sort_keys=True))
    return 0 if report["isolation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
