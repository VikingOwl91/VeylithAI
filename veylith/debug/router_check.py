#!/usr/bin/env python3
"""Router load distribution + weight sanity for the phase4 trained checkpoints."""
import sys, os
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/../v2")
import sparse_moe_v2_train as sparse_impl
import train as dense_impl

device = "cuda"
for label, mod in (("sparse", sparse_impl), ("dense", dense_impl)):
    ckpt = torch.load(f"phase4/{'sparse' if label=='sparse' else 'dense'}-run/checkpoints/ckpt_{label}.pt",
                      map_location="cpu", weights_only=False)
    cfg = mod.Config(**ckpt["config"])
    model = mod.GPT(cfg); model.load_state_dict(ckpt["model"]); model.to(device); model.eval()
    nonfinite = sum(int((~torch.isfinite(p)).sum()) for p in model.parameters())
    print(f"[{label}] val_loss={ckpt['val_loss']:.4f} | nonfinite params: {nonfinite}")
    data = torch.randint(0, cfg.vocab_size, (100000,))
    for trial in range(3):
        x, y = mod.get_batch(data, cfg.block_size, 64, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = model.tok_emb(x)
            for li, block in enumerate(model.blocks):
                h = h + block.attn(block.ln1(h))
                mlp_in = block.ln2(h)
                if getattr(block, "is_moe", False):
                    rl = block.mlp.gate(mlp_in.reshape(-1, mlp_in.size(-1))).float()
                    probs = F.softmax(rl, -1).mean(0)
                    if trial == 2:
                        print(f"  L{li}: {[f'{p:.3f}' for p in probs.tolist()]} | spread={probs.max()-probs.min():.3f}")
                out_mlp = block.mlp(mlp_in)
                mlp_out = out_mlp[0] if isinstance(out_mlp, tuple) else out_mlp
                h = h + mlp_out
    print()
