# Reproducing the v3 data pipeline

All commands are CPU-only and run from the repository root. No GPU, no
system package, and no network for the test suite.

## Environment

| item | value |
|---|---|
| Python | 3.14.8 |
| numpy | 2.5.4 |
| sentencepiece | 0.2.2 |
| `datasets` | not installed (optional scale backend) |
| network | only for the FineWeb-Edu smoke and materialization |

## Tests

```bash
python3 -m pytest veylith/v3/tests -q           # 36 passed, 1 skipped
python3 -m pytest veylith/v2/tests veylith/tests veylith/v3/tests -q  # 125 passed, 1 skipped
```

The skipped test is the opt-in network smoke
(`VEYLITH_NETWORK_TESTS=1`). The suite uses a fake source and a local JSONL
source; it writes only to pytest temporary directories.

## Source smoke (network)

```bash
python3 veylith/v3/pipeline.py inspect --source fineweb-edu --limit 100 \
    --page-delay 2.5
```

One page, 100 documents, no write. On 2026-10-26 it returned revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, license `odc-by`, public, schema
`text, id, dump, url, file_path, language, language_score, token_count,
score, int_score`.

## 10M-token pilot

```bash
ROOT=~/veylith-v3-data/fineweb-10m-20261026

python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 11400 \
    --root "$ROOT" --shard-docs 1000 --min-chars 200 --max-chars 20000 \
    --val-permille 10 --test-permille 10 --max-rows 40000 --page-delay 2.5

python3 veylith/v3/pipeline.py tokenize --root "$ROOT" --vocab 8192
python3 veylith/v3/pipeline.py compare-tokenizers --root "$ROOT" \
    --vocabs 4096,8192
python3 veylith/v3/pipeline.py verify --root "$ROOT" --deep --near-dups
python3 veylith/v3/benchmark_loader.py --root "$ROOT" \
    --seq-len 256 --batch-size 8 --batches 200 --warmup 50 \
    --out "$ROOT/loader-benchmark.json"
```

Reference values of the pilot materialized on 2026-10-26:

| item | value |
|---|---|
| rows read | 11,754 |
| accepted / kept | 11,400 / 11,399 |
| splits | train 11,173, validation 125, test 101 |
| rejected | 354 `too_long` |
| duplicates | 1 exact, 0 near |
| characters | 40,674,730 train, 467,087 validation, 342,051 test |
| tokens | 10,333,164 train, 119,572 validation, 85,955 test (10,538,691 total) |
| chars/token | 3.9363 train |
| manifest digest | `e995fdead3a6bf57d6db5f768f19c4363515d44a9a0bdbf860539e454711dfdc` |
| tokenizer model sha256 | `537afa0a4c64690bb5131db1f3d7e1c490f378a6647c35b57c7f73e252a61df6` |
| tokenizer vocab sha256 | `4181b0c390a02fc9cd5df67df395eebbccee1ae755c1829cfa1c28f1452ddf60` |
| tokenizer train text sha256 | `f44dcf7de18d2ac7230517fbe5158db94829f38e7009d73c29c4e8ceba3c3585` |
| source revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| source license | ODC-By 1.0 |
| root size | 194 MB (staging 48 MB) |
| network + prepare | 427 s, peak 87.8 MB |
| tokenize | 10.7 s, peak 444.8 MB |
| verify deep + near-dups | 54.3 s, peak 135.3 MB |

The digest depends on the exact source rows, filters, split permille values
and shard sizes. The full manifest (source revision, license, counts, every
shard sha256, tokenizer hashes) is committed at
`veylith/v3/docs/pilot-10m-manifest-2026-10-26.json`; it contains no source
text. The tokenizer comparison is committed at
`veylith/v3/docs/tokenizer-comparison-10m-2026-10-26.json`, the loader
benchmark at `veylith/v3/docs/loader-benchmark-10m-2026-10-26.json`.

`verify --deep --near-dups` decodes every token span, strips EOS and compares
the text hash with the document shard, checks every shard sha256, the manifest
digest and split isolation, and recomputes SimHashes over all kept documents.

## Tokenizer comparison

`compare-tokenizers` trains both vocabularies on the same final train
documents (never on validation or test) and measures on held-out documents:

| metric | 4096 | 8192 |
|---|---:|---:|
| chars/token | 3.4663 | 3.9412 |
| fertility (tokens/word) | 1.7815 | 1.5668 |
| tokens/doc mean | 1032.9 | 908.4 |
| embedding + head params | 2,097,152 | 4,194,304 |
| total params, v2 shape | 27,187,456 | 29,284,608 |
| training time | 2.414 s | 2.558 s |
| peak child RSS | 444 MB | 453 MB |
| roundtrip failures | 0 | 0 |

Preliminary default: 8192. This is a data-side choice, not a model-quality
claim.

## Loader benchmark and resume

`benchmark_loader.py` builds the plan, verifies coverage, measures batches
after a warmup and checks mid-epoch resume. On the 10M pilot:

| metric | value |
|---|---:|
| plan sequences | 40,364 |
| plan tokens | 10,333,164 |
| plan build | 0.068 s |
| throughput | 2,243,848 tokens/s (1,095.6 batches/s, 8 x 256) |
| peak RSS | 79.6 MB |
| resume equality | true |

`batches --start-batch N` and `iter_batches(..., start_epoch, start_batch)`
resume deterministically; a training loop can checkpoint `(epoch, batch)`.
Preparing a corpus resumes from `staging/state.json`:

```bash
python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 11400 \
    --root "$ROOT" --page-delay 2.5 --resume
```

A resume is refused when the source identity (dataset, config, split,
revision), the filter configuration or the pipeline version changed.

## Determinism notes

- Splits, dedup decisions and document ordering do not depend on arrival
  order. Exact and near duplicates keep the smallest `doc_key`.
- The tokenizer model is byte-identical across roots and processes:
  training uses `sentence_iterator` plus `model_writer`, so no root-specific
  path is embedded. The comparison child and the final `tokenize` process
  produce the same sha256.
- The first 100 document keys of the 10M corpus equal an independent
  100-document smoke at the same revision.
- The loader order digest is identical across processes; one epoch covers
  every sequence exactly once.
- HF streaming order is not a stable contract. The pinned revision and the
  local manifest are the anchors. A second full 10M network run was not
  attempted; the earlier 2000-document pilot hit HTTP 429 on an immediate
  second run. Use 2.5 s pacing, respect `Retry-After`, and use the
  `datasets` backend for scale.

## Recreate or remove

The pilots are disposable and not in git:

```bash
rm -rf ~/veylith-v3-data/fineweb-10m-20261026
rm -rf ~/veylith-v3-data/pilot-20261026
# rerun the commands above to recreate
```

No credential, corpus shard or tokenizer is stored in the repository.
