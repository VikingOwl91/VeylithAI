# veylith v3: pretraining data pipeline

> Research status: v3.0 is pretraining data infrastructure, not a trained
> model. v1.x and v2.x are experimental research prototypes, not finished or
> production-ready language models. v3.1 (initial pretraining experiments)
> has not started. No checkpoint in this repository is a released model.

This directory is the first v3 step: a modular, streaming-capable
pretraining data pipeline, CPU only, separate from GPU training. It
materializes a versioned local corpus (documents, manifests, tokenizer and
token shards) from Hugging Face sources and feeds deterministic training
batches. The model architecture is not part of this step; v2 remains the
reference.

See `PLAN.md` for goals and milestones, `docs/DATA-SOURCES.md` for sources
and licenses, `docs/REPRODUCE.md` for commands, and
`STATUS-2026-10-26.md` for the current status and test report.

## Components

| module | responsibility |
|---|---|
| `data_pipeline/sources.py` | dataset adapters, bounded streaming, revision pinning, typed errors |
| `data_pipeline/prepare.py` | normalization, filtering, deterministic split assignment, exact and near deduplication |
| `data_pipeline/dataset.py` | document shards, manifests, atomic writes, integrity verification |
| `data_pipeline/tokenize.py` | BPE training on the train split, token shards with document offsets |
| `data_pipeline/loader.py` | non-overlapping sequence packing, deterministic epoch sampling |
| `data_pipeline/cli.py` | CLI for inspect, prepare, tokenize, verify and batches |
| `pipeline.py` | entry point, `python3 veylith/v3/pipeline.py --help` |

## Guarantees

- **Streaming, bounded memory.** The source is read page by page (100 rows
  per request) or as a `datasets` iterable; no full source download. Text is
  written to staging shards, only metadata stays in memory.
- **Pinned source.** Every manifest records the dataset id, config, split,
  commit revision, license and access mode. The revision is resolved once
  and used for every request.
- **Deterministic splits.** `split = f(sha256(doc_key))`: test < 5 permille,
  validation < 10 permille, else train. A document's split does not depend
  on stream order.
- **No leakage.** Splits are assigned per document, never per sequence.
  Duplicates are removed before the split is applied. The tokenizer is
  trained only on final train-split documents.
- **Deterministic dedup.** Exact duplicates use `sha256(normalized_text)`.
  Near duplicates use 64-bit SimHash over word 5-grams, 4 bands of 16 bits,
  Hamming distance <= 3. The kept document of a cluster is the one with the
  smallest `doc_key`, independent of stream order.
- **Atomic and verifiable.** Every shard and manifest is written to a
  temporary file, fsynced and renamed. Each shard has a sha256; the manifest
  has a digest over all fields and is the single commit point.
- **No token reuse.** The concat packer cuts the token stream into
  non-overlapping windows; one epoch uses every sequence exactly once. The
  document packer never merges two documents into one window.
- **No secrets.** Public sources need no credential. The pipeline never
  reads, writes or logs authentication tokens; gated datasets use Hugging
  Face's own login store outside this repository.

## Quickstart

```bash
# source metadata and schema (no download)
python3 veylith/v3/pipeline.py source-info --source fineweb-edu

# bounded streaming smoke: 20 documents, one page, nothing written
python3 veylith/v3/pipeline.py inspect --source fineweb-edu --limit 20

# materialize 2000 documents (outside the repository)
python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 2000 \
    --root ~/veylith-v3-data/pilot --shard-docs 500

# train the 8192 BPE tokenizer on the train split, write token shards
python3 veylith/v3/pipeline.py tokenize --root ~/veylith-v3-data/pilot --vocab 8192

# verify shard hashes, split isolation and every decoded document span
python3 veylith/v3/pipeline.py verify --root ~/veylith-v3-data/pilot --deep

# deterministic batch sample
python3 veylith/v3/pipeline.py batches --root ~/veylith-v3-data/pilot \
    --seq-len 256 --batch-size 4 --batches 2
```

Exit codes: 0 success, 1 validation or local error, 2 access denied,
3 network failure, 4 source or configuration error.

## Data layout

```
<root>/
  staging/
    state.json                 resumable source pass (offset, counters)
    index-00000.jsonl          hashes and counts per accepted document
    shard-00000.jsonl          normalized documents, arrival order
  corpus/
    manifest.json              the commit point; digest over all fields
    documents/{train,validation,test}/*.jsonl
    tokenizer/bpe8192.{model,vocab,train.txt}
    tokens/{train,validation,test}/*.tokens.npy
    tokens/{train,validation,test}/*.meta.json
```

The default root is `veylith/v3/data` and is gitignored. The pilot lives
outside the repository; no corpus, tokenizer or token shard is committed.

## Sources

Implemented: FineWeb-Edu, config `sample-10BT`, public, ODC-By 1.0. Two
backends:

- `http` (default): paged rows from the Hugging Face datasets-server, 100
  rows per request, bounded retries, `revision` pinned. No extra dependency.
- `datasets`: `datasets.load_dataset(..., streaming=True)`. The scale path;
  requires the library in a project environment. It is not installed in the
  system Python, so the pipeline does not depend on it.

Documented extension points, not implemented: Wikimedia Wikipedia,
Cosmopedia v2, Python-Edu. See `docs/DATA-SOURCES.md`.

## Tokenizer

BPE, 8192 tokens, byte fallback, trained on the train split only. Special
tokens: `<unk>` 0, `<s>` 1, `</s>` 2, `<pad>` 3. Each document ends with
`</s>`; per-document token offsets are stored next to the array. Identity
normalization, no dummy prefix, deterministic training
(`num_threads=1`, `shuffle_input_sentence=False`, fixed document order).
The training text is written with `sentence_iterator`/`model_writer`, so the
model bytes do not embed a root-specific path.

Parameter effect for the v2-shaped reference model: the untied token
embedding and output head grow by `(8192 - 1000) * 256 * 2 = 3,682,304`
parameters, so about 29.28M total instead of 25.60M, with the same 9.89M
active per token.

## Tests

```bash
python3 -m pytest veylith/v3/tests -q
```

The suite is CPU-only and offline. It covers normalization and filters,
split determinism and isolation, exact and near deduplication, shard and
manifest integrity, resume, tokenizer reproducibility and train-only
training, token roundtrips and document boundaries, packing coverage and
sampling determinism, source error classes and the CLI end to end. The
network smoke test is opt-in: `VEYLITH_NETWORK_TESTS=1`.

## Limits

- The prepare step holds the index metadata of all staged documents in
  memory (about 250 bytes per document). Fine for the 2k pilot; before 10M
  documents it needs bucketing or external sorting.
- Near-duplicate detection targets shared boilerplate and near-identical
  text. Small edits in otherwise unique text can exceed the Hamming
  threshold; MinHash or embedding-based dedup is future work.
- The paged HTTP backend is rate limited by the datasets-server. A second
  full 2000-document materialization hit HTTP 429 after three bounded
  retries. The `datasets` backend is the scale path; the local materialized
  corpus is the reproducibility anchor.
- The tokenizer plan (8192 BPE) is a decision for v3; it is not a claim
  about the best vocabulary for the final corpus.
- No GPU code, no model training, no benchmark in this step.
