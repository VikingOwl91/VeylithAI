# Reproducing the v3 data pipeline

All commands are CPU-only and run from the repository root. No GPU, no
system package, and no network for the test suite.

## Environment

| item | value |
|---|---|
| project env | `~/veylith-v3-env`, Python 3.12.12 |
| datasets / pyarrow | 5.1.0 / 26.0.0 (project env only) |
| numpy / sentencepiece | 2.5.4 / 0.2.2 |
| system Python | 3.14.8, used for the v2/torch tests |
| network | only for the FineWeb-Edu/Wikipedia/Cosmopedia smoke and materialization |

Create the isolated environment once (no global packages):

```bash
uv venv --python 3.12 ~/veylith-v3-env
uv pip install --python ~/veylith-v3-env/bin/python \
    datasets pyarrow sentencepiece numpy pytest
```

## Tests

```bash
~/veylith-v3-env/bin/python -m pytest veylith/v3/tests -q
# 43 passed, 3 skipped in the project env (torch absent skips architecture)
python3 -m pytest veylith/v3/tests -q
# 62 passed, 1 skipped on system Python (architecture tests included)
python3 -m pytest veylith/v2/tests veylith/tests -q   # 89 passed
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

## Multi-source 100M corpus

Requires the project environment (datasets/pyarrow). The plan files live in
`veylith/v3/plans/`.

```bash
VENV=~/veylith-v3-env/bin/python
ROOT=~/veylith-v3-data/multi-100m-20261026

$VENV veylith/v3/pipeline.py prepare-multi \
    --plan veylith/v3/plans/corpus-100m.json --root "$ROOT"
$VENV veylith/v3/pipeline.py tokenize --root "$ROOT" --vocab 8192
$VENV veylith/v3/pipeline.py verify --root "$ROOT" --deep --near-dups
$VENV veylith/v3/pipeline.py report-multi --root "$ROOT"
$VENV veylith/v3/benchmark_loader.py --root "$ROOT" \
    --seq-len 256 --batch-size 8 --batches 200 --warmup 50
```

Reference values (2026-10-26):

| item | value |
|---|---|
| sources used | FineWeb-Edu, Wikipedia, Cosmopedia v2 |
| blocked | Python-Edu (The Stack v2 gated, license `other`, no text) |
| revisions | `87f0914`, `b04c8d1c`, `3ba9d605` |
| rows / accepted / available | 97,280 / 95,188 / 95,180 |
| selected documents | 90,784 (train 87,063, validation 1,874, test 1,847) |
| train tokens | 95,951,233 (60,356,799 / 20,335,474 / 15,258,960) |
| validation / test tokens | 2,122,377 / 2,094,292 |
| tokenizer model sha256 | `411227a8e5289d6f4b450059b744ed278532a4624d3895f1a4979600c49ae034` |
| manifest digest | `7a6e7991dd83244ced4c9c13627028e1824f00b0f58735cc665323ff111bdcf0` |
| root size | 1.9 GB |
| prepare / tokenize / verify | 811 s / 98 s / 375 s |
| peak RSS | 2.78 / 3.68 / 0.91 GB |
| loader | 374,810 sequences, 1.10M tokens/s, 227 MB, resume equal |

The pilot plan is `veylith/v3/plans/pilot-multi-1m.json`; it produces about
1M train tokens per cleared source. The blocked source is listed in the plan
and logged; it is not silently skipped.

Per-source held-out fertility of the 8192 tokenizer: FineWeb 1.5533,
Wikipedia 1.7487, Cosmopedia 1.5668. The manifest and the multi-source
report are committed under `veylith/v3/docs/`.

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
