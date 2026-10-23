#!/usr/bin/env python3
"""Generate text from a trained checkpoint.

Usage:
    python sample.py                          # default prompt, 500 tokens
    python sample.py --prompt "JULIET:" --temperature 1.0 --max_new_tokens 800
"""

import argparse

import torch

from train import GPT, Config, generate, pick_device, tokenizer_from_info


def main():
    p = argparse.ArgumentParser(description="sample from a trained checkpoint")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    p.add_argument("--prompt", type=str, default="ROMEO:")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    # weights_only=False because the checkpoint stores tokenizer dicts
    # alongside the weights. It is a file we saved ourselves.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GPT(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    info = ckpt.get("tokenizer")
    if info is None:  # checkpoint from before the tokenizer upgrade
        info = {"type": "char", "stoi": ckpt["stoi"], "itos": ckpt["itos"]}
    encode, decode = tokenizer_from_info(info)

    print(f"loaded {args.checkpoint} (val loss {ckpt['val_loss']:.4f}) on {device}\n")
    print(generate(model, encode, decode, device,
                   prompt=args.prompt,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature,
                   top_k=args.top_k))


if __name__ == "__main__":
    main()
