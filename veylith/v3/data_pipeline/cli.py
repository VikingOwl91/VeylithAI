"""CLI for the v3 data pipeline.

Examples:

    python3 veylith/v3/pipeline.py source-info --source fineweb-edu
    python3 veylith/v3/pipeline.py inspect --source fineweb-edu --limit 20
    python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 2000 \\
        --root ~/veylith-v3-data/pilot
    python3 veylith/v3/pipeline.py tokenize --root ~/veylith-v3-data/pilot
    python3 veylith/v3/pipeline.py verify --root ~/veylith-v3-data/pilot --deep
    python3 veylith/v3/pipeline.py batches --root ~/veylith-v3-data/pilot

Exit codes: 0 success, 1 validation or local error, 2 access denied,
3 network failure, 4 source or configuration error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import dataset as ds
from . import loader, prepare, sources, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
DEFAULT_ROOT = os.path.join(V3, "data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="veylith v3 pretraining data pipeline (CPU only)")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("source-info", help="show source spec and hub metadata")
    info.add_argument("--source", default="fineweb-edu")
    info.add_argument("--revision", default=None)
    info.set_defaults(func=cmd_source_info)

    inspect = sub.add_parser("inspect", help="stream a few documents, write nothing")
    inspect.add_argument("--source", default="fineweb-edu")
    inspect.add_argument("--config", default=None)
    inspect.add_argument("--revision", default=None)
    inspect.add_argument("--backend", default="auto", choices=["auto", "http", "datasets"])
    inspect.add_argument("--path", default=None, help="local-jsonl source file")
    inspect.add_argument("--limit", type=int, default=10)
    inspect.add_argument("--chars", type=int, default=100, help="preview characters")
    inspect.set_defaults(func=cmd_inspect)

    prep = sub.add_parser("prepare", help="stream, filter, dedup and write document shards")
    prep.add_argument("--source", default="fineweb-edu")
    prep.add_argument("--config", default=None)
    prep.add_argument("--revision", default=None)
    prep.add_argument("--backend", default="auto", choices=["auto", "http", "datasets"])
    prep.add_argument("--path", default=None, help="local-jsonl source file")
    prep.add_argument("--root", default=DEFAULT_ROOT)
    prep.add_argument("--docs", type=int, required=True, help="target kept documents")
    prep.add_argument("--shard-docs", type=int, default=1000)
    prep.add_argument("--min-chars", type=int, default=200)
    prep.add_argument("--max-chars", type=int, default=20000)
    prep.add_argument("--val-permille", type=int, default=5)
    prep.add_argument("--test-permille", type=int, default=5)
    prep.add_argument("--near-dup-hamming", type=int, default=3)
    prep.add_argument("--max-rows", type=int, default=None)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--resume", action="store_true")
    prep.set_defaults(func=cmd_prepare)

    tok = sub.add_parser("tokenize", help="train BPE on train docs and write token shards")
    tok.add_argument("--root", default=DEFAULT_ROOT)
    tok.add_argument("--vocab", type=int, default=8192)
    tok.set_defaults(func=cmd_tokenize)

    verify = sub.add_parser("verify", help="verify hashes, splits and tokenizer links")
    verify.add_argument("--root", default=DEFAULT_ROOT)
    verify.add_argument("--deep", action="store_true",
                        help="decode every token span and compare text hashes")
    verify.set_defaults(func=cmd_verify)

    batches = sub.add_parser("batches", help="print deterministic batch metadata")
    batches.add_argument("--root", default=DEFAULT_ROOT)
    batches.add_argument("--split", default="train", choices=list(ds.SPLITS))
    batches.add_argument("--seq-len", type=int, default=256)
    batches.add_argument("--batch-size", type=int, default=4)
    batches.add_argument("--batches", type=int, default=2)
    batches.add_argument("--epochs", type=int, default=1)
    batches.add_argument("--seed", type=int, default=0)
    batches.add_argument("--mode", default="concat", choices=["concat", "document"])
    batches.set_defaults(func=cmd_batches)
    return parser


def _open(args):
    return sources.open_source(
        args.source, config=args.config, revision=args.revision,
        backend=args.backend, path=getattr(args, "path", None))


def cmd_source_info(args) -> int:
    spec = sources.source_spec(args.source)
    payload = {"spec": spec.__dict__}
    if args.source == "fineweb-edu":
        payload["hub"] = sources.fetch_dataset_info(spec.dataset_id)
    payload["planned"] = sorted(sources.PLANNED_SOURCES)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_inspect(args) -> int:
    if not 1 <= args.limit <= 1000:
        raise ValueError("--limit must be between 1 and 1000")
    source = _open(args)
    for count, doc in enumerate(source.iter_documents(limit_rows=args.limit), start=1):
        preview = " ".join(doc.text.split())[:args.chars]
        print(f"[{count}] {doc.doc_key}")
        print(f"    chars={len(doc.text)} language={doc.meta.get('language')} "
              f"int_score={doc.meta.get('int_score')} "
              f"row={doc.meta.get('row_index')}")
        print(f"    {preview!r}")
    return 0


def cmd_prepare(args) -> int:
    if args.docs <= 0:
        raise ValueError("--docs must be positive")
    source = _open(args)
    cfg = prepare.FilterConfig(min_chars=args.min_chars, max_chars=args.max_chars)
    manifest = prepare.run_prepare(
        source, args.root, limit_docs=args.docs, filter_config=cfg,
        val_permille=args.val_permille, test_permille=args.test_permille,
        shard_docs=args.shard_docs, resume=args.resume, max_rows=args.max_rows,
        near_dup_hamming=args.near_dup_hamming, seed=args.seed)
    counts = manifest["counts"]
    print(f"manifest: {ds.manifest_path(args.root)}")
    print(f"digest:   {manifest['digest']}")
    print(f"source:   {manifest['source']['dataset_id']}@{manifest['source']['revision']} "
          f"({manifest['source']['license']})")
    print(f"documents: train={counts['documents']['train']} "
          f"validation={counts['documents']['validation']} "
          f"test={counts['documents']['test']}")
    print(f"accepted={counts['accepted']} exact_duplicates={counts['exact_duplicates']} "
          f"near_duplicates={counts['near_duplicates']}")
    print(f"rejected={counts['rejected']}")
    return 0


def cmd_tokenize(args) -> int:
    tokenization = tokenize.tokenize_corpus(args.root, vocab_size=args.vocab)
    print(f"tokenizer: {tokenization['tokenizer']['model']} "
          f"sha256={tokenization['tokenizer']['sha256']}")
    for split in ds.SPLITS:
        entry = tokenization["splits"][split]
        print(f"  {split}: {entry['docs']} docs, {entry['tokens']:,} tokens")
    print(f"total tokens: {tokenization['total_tokens']:,}")
    return 0


def cmd_verify(args) -> int:
    result = ds.verify_corpus(args.root, deep=args.deep)
    if result["ok"]:
        print(f"ok: digest={result['digest']} counts={json.dumps(result['counts'])}")
        return 0
    print("FAILED:")
    for problem in result["problems"]:
        print(f"  - {problem}")
    return 1


def cmd_batches(args) -> int:
    plan = loader.plan_sequences(args.root, split=args.split, seq_len=args.seq_len,
                                 mode=args.mode)
    problems = loader.verify_plan(plan)
    if problems:
        print("plan is invalid:", problems, file=sys.stderr)
        return 1
    print(f"plan: {len(plan['sequences'])} sequences, {plan['tokens']} tokens, "
          f"mode={plan['mode']}, seq_len={plan['seq_len']}")
    for batch in loader.iter_batches(plan, args.root, batch_size=args.batch_size,
                                     epochs=args.epochs, seed=args.seed,
                                     max_batches=args.batches):
        print(f"epoch={batch['epoch']} batch={batch['batch']} "
              f"indices={batch['indices']} lengths={batch['lengths']} "
              f"order={batch['order_digest'][:16]}")
        for row in batch["tokens"]:
            print("   ", row[:16].tolist())
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except sources.SourceAuthError as exc:
        print(f"access denied: {exc}", file=sys.stderr)
        return 2
    except sources.SourceNetworkError as exc:
        print(f"network failure: {exc}", file=sys.stderr)
        return 3
    except sources.SourceError as exc:
        print(f"source error: {exc}", file=sys.stderr)
        return 4
    except (ValueError, OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
