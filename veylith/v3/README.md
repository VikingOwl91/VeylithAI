# veylith v3: pretraining data pipeline

> Research status: v3.0 is pretraining data infrastructure, not a trained
> model. v1.x and v2.x are experimental research prototypes, not finished or
> production-ready language models. v3.1-A (initial pretraining
> experiments) has a CPU-validated architecture design; no training run
> has started. No checkpoint in this repository is a released model.

This directory is the first v3 step: a modular, streaming-capable
pretraining data pipeline, CPU only, separate from GPU training. It
materializes a versioned local corpus (documents, manifests, tokenizer and
token shards) from Hugging Face sources and feeds deterministic training
batches. The v3.1-A architecture is designed and CPU-validated in
`architecture/`; v2 remains the frozen reference and no training run has
started.

The v3.0 corpus is materialized: 90,784 documents from FineWeb-Edu,
Wikipedia and Cosmopedia v2, 95,951,233 train tokens, 8192-token BPE, 2 %
validation and 2 % test, deep verification green. Python-Edu is blocked
because its text requires The Stack v2 (gated, license `other`); the 5 %
share is reported, not backfilled. See
`docs/MULTI-SOURCE-100M-2026-10-26.md`, the earlier
`docs/PILOT-10M-2026-10-26.md` and `docs/V3.1-DESIGN.md`.

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
| `data_pipeline/loader.py` | non-overlapping sequence packing, deterministic epoch sampling, mid-epoch resume |
| `data_pipeline/tokenizer_compare.py` | train and compare BPE vocabularies on the train split |
| `data_pipeline/multi_source.py` | per-source plans, cross-source merge, quotas, bucket shuffle, per-source report |
| `architecture/config.py` | v3.1-A candidates and validation |
| `architecture/model.py` | CPU reference model: GQA/RoPE, hybrid SWA, sparse/dense MoE, no MTP |
| `architecture/accounting.py` | exact parameter accounting, active counts, memory and compute estimates |
| `data_pipeline/cli.py` | CLI for inspect, prepare, prepare-multi, merge, tokenize, compare, verify, report and batches |
| `plans/` | reproducible multi-source budget plans (pilot and 100M) |
| `benchmark_loader.py` | CPU loader throughput and peak-RSS benchmark |
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
  document packer never merges two documents into one window. The batch
  cache is bounded to 8 token shards.
- **Resumable loader.** `iter_batches(..., start_epoch, start_batch)` skips
  deterministically to a saved position; a training loop can checkpoint
  `(epoch, batch)` after every step.
- **No secrets.** Public sources need no credential. The pipeline never
  reads, writes or logs authentication tokens; gated datasets use Hugging
  Face's own login store outside this repository.

## Quickstart

```bash
# source metadata and schema (no download)
python3 veylith/v3/pipeline.py source-info --source fineweb-edu

# bounded streaming smoke: 100 documents, one page, nothing written
python3 veylith/v3/pipeline.py inspect --source fineweb-edu --limit 100 \
    --page-delay 2.5

# materialize the 10M-token pilot (outside the repository)
python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 11400 \
    --root ~/veylith-v3-data/fineweb-10m-20261026 --shard-docs 1000 \
    --val-permille 10 --test-permille 10 --page-delay 2.5

# train the 8192 BPE tokenizer on the train split, write token shards
python3 veylith/v3/pipeline.py tokenize \
    --root ~/veylith-v3-data/fineweb-10m-20261026 --vocab 8192

# train and compare 4096 against 8192 on the same train split
python3 veylith/v3/pipeline.py compare-tokenizers \
    --root ~/veylith-v3-data/fineweb-10m-20261026 --vocabs 4096,8192

# verify shard hashes, splits, decoded spans and near-duplicate isolation
python3 veylith/v3/pipeline.py verify \
    --root ~/veylith-v3-data/fineweb-10m-20261026 --deep --near-dups

# deterministic batch sample and mid-epoch resume
python3 veylith/v3/pipeline.py batches \
    --root ~/veylith-v3-data/fineweb-10m-20261026 \
    --seq-len 256 --batch-size 8 --batches 2
python3 veylith/v3/pipeline.py batches \
    --root ~/veylith-v3-data/fineweb-10m-20261026 \
    --seq-len 256 --batch-size 8 --batches 2 --start-batch 2

# loader throughput and peak RSS
python3 veylith/v3/benchmark_loader.py \
    --root ~/veylith-v3-data/fineweb-10m-20261026

# multi-source pipeline (project environment with datasets/pyarrow)
python3 veylith/v3/pipeline.py prepare-multi \
    --plan veylith/v3/plans/pilot-multi-1m.json \
    --root ~/veylith-v3-data/multi-pilot-20261026
python3 veylith/v3/pipeline.py prepare-multi \
    --plan veylith/v3/plans/corpus-100m.json \
    --root ~/veylith-v3-data/multi-100m-20261026
python3 veylith/v3/pipeline.py tokenize \
    --root ~/veylith-v3-data/multi-100m-20261026 --vocab 8192
python3 veylith/v3/pipeline.py verify \
    --root ~/veylith-v3-data/multi-100m-20261026 --deep --near-dups
python3 veylith/v3/pipeline.py report-multi \
    --root ~/veylith-v3-data/multi-100m-20261026
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

| source | config | license | status |
|---|---|---|---|
| FineWeb-Edu | `sample-10BT` | ODC-By 1.0 | used |
| Wikipedia | `20231101.en` | CC BY-SA 3.0 + GFDL | used |
| Cosmopedia v2 | `cosmopedia-v2` (SmolLM-Corpus) | ODC-By 1.0 | used, synthetic |
| Python-Edu | `python-edu` (SmolLM-Corpus) | The Stack v2, `other` | **blocked** (no text, gated terms) |

Two backends:

- `http`: paged rows from the Hugging Face datasets-server, 100 rows per
  request, bounded retries, `revision` pinned. No extra dependency.
- `datasets`: `datasets.load_dataset(..., streaming=True)` in the project
  environment (`~/veylith-v3-env`, Python 3.12, datasets 5.1.0,
  pyarrow 26.0.0). The multi-source runs use this backend.

See `docs/DATA-SOURCES.md` for revisions, schemas, attribution and the
Python-Edu block.

## Tokenizer

BPE, 8192 tokens (preliminary default), byte fallback, trained on the train
split only. Special tokens: `<unk>` 0, `<s>` 1, `</s>` 2, `<pad>` 3. Each
document ends with `</s>`; per-document token offsets are stored next to the
array. Identity normalization, no dummy prefix, deterministic training
(`num_threads=1`, `shuffle_input_sentence=False`, fixed document order).
The training text is written with `sentence_iterator`/`model_writer`, so the
model bytes do not embed a root-specific path.

Measured on 226 held-out documents of the 10M pilot:

| metric | 4096 | 8192 |
|---|---:|---:|
| chars/token | 3.4663 | 3.9412 |
| fertility (tokens/word) | 1.7815 | 1.5668 |
| tokens/doc mean | 1032.9 | 908.4 |
| embedding+head params | 2,097,152 | 4,194,304 |
| total params, v2 shape | 27,187,456 | 29,284,608 |
| training time / peak RSS | 2.414 s / 444 MB | 2.558 s / 453 MB |
| roundtrip failures | 0 | 0 |

8192 is chosen for the 12 % lower fertility at 2.1M extra parameters; no
model quality is claimed.

Multi-source held-out fertility (100M corpus, 8192):

| source | heldout docs | chars/token | fertility |
|---|---:|---:|---:|
| FineWeb-Edu | 2,692 | 3.9458 | 1.5533 |
| Wikipedia | 319 | 3.6082 | 1.7487 |
| Cosmopedia v2 | 710 | 4.3743 | 1.5668 |

## v3.1-A architecture

Four ~0.1B candidates are defined; the primary is A.

| id | config | total | active (sparse) | role |
|---|---|---:|---:|---|
| A | 8L, 512w, GQA 8/2, top-2, ffn 960, no shared | 98.34M | 36.41M | primary |
| B | 10L, 512w, top-2, ffn 800, 2 dense | 97.82M | 38.84M | depth ablation |
| C | 12L, 384w, top-2, ffn 736, shared | 95.88M | 39.92M | fallback |
| D | 8L, 512w, top-1, ffn 896, shared | 102.47M | 35.03M | routing ablation |

Vocabulary 8192, context 512, no MTP. Counts are exact for the analytical
formulas and cross-checked on meta-instantiated models; active counts are
cross-checked with forward hooks. Memory and compute estimates, the MTP
exclusion and the evaluation protocol are in
`docs/V3.1-ARCHITECTURE.md`, `docs/V3.1-MEMORY-COMPUTE.md` and
`docs/V3.1-EVALUATION.md`.

## v3.1-B training infrastructure

CPU reference trainer for architecture A: next-token CE with optional router
aux (no MTP), AdamW and cosine warmup, gradient accumulation and clipping,
deterministic seeds, JSONL metrics, per-source evaluation (CE, bits/char,
router, train/val gap) and atomic checkpoints with model/optimizer/schedule/
RNG/loader state and validation-only selection. SDPA and manual attention,
sparse and dense-combined MoE paths are parity-tested. A guard refuses any
CPU run above 5M parameters; no 0.1B CPU run and no GPU run happened.
Docs: `docs/V3.1-TRAINING.md`, `docs/V3.1-CHECKPOINTS.md`,
`docs/V3.1-GPU-SMOKE-PLAN.md` (proposal only).

The authorized GPU compatibility smoke ran on 2026-10-27: bf16 baseline,
SDPA parity (fp32 exact, bf16 max abs 0.003906, identical input grads),
sparse top-2 MoE at A width, and full A at microbatch 1/2/4/8 all passed in
8.3 s. MB8: 0.150 s/step, 27,345 tokens/s, 3.40 GB allocated / 3.58 GB
reserved, no fault. No pretraining was started. Results:
`docs/V3.1-GPU-SMOKE-RESULTS.md`.

## Tests

```bash
python3 -m pytest veylith/v3/tests -q
```

The suite is CPU-only and offline. On the system Python: 71 passed,
1 skipped (network smoke); in the project environment: 70 passed,
2 skipped (network and the missing-`datasets` path). The v2 suite is
89 passed. Combined system run: 160 passed, 1 skipped. Covered: normalization and filters, split determinism and
isolation, exact and near deduplication (including across sources), quotas
and shortfall, shard and manifest integrity, resume, tokenizer
reproducibility and train-only training, tokenizer comparison, token
roundtrips and document boundaries, packing coverage, mid-epoch resume and
sampling determinism, source error classes and the CLI end to end. The
network smoke test is opt-in: `VEYLITH_NETWORK_TESTS=1`.

## Limits

- The prepare step holds the index metadata of all staged documents in
  memory and on disk (about 320 bytes per document on disk). The 100M
  multi-source run peaked at 2.78 GB RSS for 90,784 documents (adapters,
  datasets buffers and the merge pass included). 10M documents need
  bucketing or external sorting.
- Near-duplicate detection targets shared boilerplate and near-identical
  text. Small edits in otherwise unique text can exceed the Hamming
  threshold; MinHash or embedding-based dedup is future work.
- The paged HTTP backend is rate limited by the datasets-server. The 10M
  pilot used 2.5 s pacing over 119 pages without a 429; an immediate second
  full 2000-document run hit HTTP 429 after three bounded retries. The
  `datasets` backend is the scale path; the local materialized corpus is the
  reproducibility anchor.
- Validation and test in the 100M corpus have 1,874 and 1,847 documents
  (2.06 % / 2.03 %) with all three sources represented. Enough for
  tokenizer and per-source evaluation, not for broad quality claims.
- Python-Edu is blocked; the 100M target is 95.95M train tokens. The gap is
  reported, not backfilled.
- Cosmopedia is 15.9 % synthetic; contamination and hallucination risks are
  documented in the multi-source report and the v3.1 design note.
- No GPU code, no model training, no model benchmark in this step.


The first real-data training run (v3.1-D) completed 100 updates on the
verified corpus: validation macro main CE 9.1168 -> 7.4325, bits/char
3.3452 -> 2.7085, all three sources improved, router healthy, 36.5k
tokens/s, 1.77 GB allocated / 4.12 GB reserved, no fault. See
`docs/V3.1-FIRST-RUN-RESULTS.md`; checkpoints stay outside Git. The v3.1-D
bits/char values use the older proportional document-share estimate; they
are not directly comparable to the exact prefix-decode values below.

The v3.1-E single-pass attempt (commit `b97a73d`) implemented the exact
prefix-decode bits/char protocol, saved the step-0 state and the initial
validation, and trained through at least 60 updates with stable loss and
routing (main CE 8.85 -> 7.42, grad norm 1.93 -> 0.27, router entropy
0.89 -> 0.99, 36.4k tokens/s, 1.78 GB allocated / 4.04 GB reserved). It then
aborted with a GPU memory access fault at about 115 s, before the step-100
checkpoint. There was no retry, no dense fallback, no test evaluation and no
further training. See `docs/V3.1-E-PARTIAL-RUN-RESULTS.md` and
`docs/V3.1-FULL-RUN-PLAN.md`.

The isolated GPU training environment (v3.1-H) is at
`~/veylith-v31-isolated`. Its pinned lock, launcher, fingerprint baseline and
read-only drift guard are under `env/`; the CPU suite in that environment
passes 97 tests with 1 skip. The isolated GPU smoke (v3.1-I) then passed on
gfx1201: BF16, SDPA
causal and sliding-window attention, Dense-Combined MoE, and a checkpoint
forward, with 0 `/opt/rocm` libraries, peak allocation 0.7445 GB and 6.7 s
total. See `docs/V3.1-I-ISOLATED-GPU-SMOKE.md`. Training remains pending
separate authorization. The isolated Sparse-MoE reproducer (v3.1-J) then
reproduced the historical GPU memory access fault with the same 2 MiB-aligned
signature and exit 134; Python and ROCm user-space isolation did not remove
it. See `docs/V3.1-J-SPARSE-REPRODUCER.md`.

The first isolated Dense-Combined training attempt (v3.1-K) stopped after 3
optimizer updates on the 6 GiB peak-reserved-VRAM guard at 6.105 GiB. It was
not a GPU fault or OOM, but it produced no checkpoint, no final validation and
no stability result. See `docs/V3.1-K-DENSE-RUN.md`.

The v3.1-L run raised that guard to 12 GiB and completed the full 500-update
horizon with exit 0 in 511 s. Peak reserved memory was 6.107 GiB, so the
v3.1-K limit sat just below the allocator's steady state. Validation macro CE
fell from 9.1168 to 5.4001, seven step-suffixed states were written and
structurally restored, and no GPU fault occurred. Router balance is healthy
but close to uniform. See `docs/V3.1-L-DENSE-RUN.md`.
