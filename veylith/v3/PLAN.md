# veylith v3: data pipeline plan

Status 2026-10-26. Branch `v3/data-pipeline`, based on the frozen v2 tag
`v2-research-freeze-2026-10-26`. This document is the plan for the first v3
step: a modular, streaming-capable pretraining data pipeline. It does not
change the v2 baseline.

## 1. Why data first

The P1-3 run showed v2 overfitting after 500 steps on 0.44M training
tokens: 25.6M parameters, 166 corpus passes in 5000 steps, validation CE
rising from 3.28 to 4.75 while training CE fell to 0.59. Changing the
architecture again before the corpus changes would repeat the same
confound. v3 therefore starts with data volume, diversity and a clean,
versioned pipeline; the model stays at the v2 architecture for the first
data-only iteration.

## 2. Goals

1. Single-pass pretraining on a much larger, more diverse corpus than Tiny
   Shakespeare.
2. Hugging-Face streaming for acquisition, without a full source download.
3. Locally materialized, versioned document and token shards with manifests.
4. Strict separation of data preparation (CPU only) from GPU training.
5. Reproducible splits, tokenization and training order.
6. CPU-only validation before any GPU approval.

## 3. Non-goals for this step

- No model implementation or architecture change; v2 remains the reference.
- No GPU training, benchmarking or fault reproduction.
- No changes to ROCm, PyTorch, the kernel, drivers or system packages.
- No full 10M/100M-token download in this step.
- No remote push, no dataset or checkpoint publication.
- No secrets, no auth tokens in files, logs or manifests.

## 4. Pipeline architecture

Five components with one responsibility each, importable without a GPU:

| component | module | responsibility |
|---|---|---|
| sources | `data_pipeline/sources.py` | dataset adapters, streaming, revision pinning, schema and license metadata |
| prepare | `data_pipeline/prepare.py` | normalization, filtering, deterministic split assignment, exact and near dedup |
| dataset | `data_pipeline/dataset.py` | document shards, manifests, atomic writes, integrity verification |
| tokenize | `data_pipeline/tokenize.py` | BPE training on the train split only, token shards with document offsets |
| loader | `data_pipeline/loader.py` | non-overlapping sequence packing, deterministic sampling |

The CLI wraps them: `python3 veylith/v3/pipeline.py <command>`.

### Data flow

```
HF stream (pinned revision)
  -> normalize + filter (CPU, bounded memory)
  -> staging shards + index (resumable)
  -> deterministic dedup + split assignment
  -> final document shards (train/validation/test)
  -> BPE training on train only
  -> token shards with per-document offsets
  -> sequence plan + deterministic batches
```

### Reproducibility contract

- Source revision is pinned to a commit sha in every manifest.
- Split assignment is a pure function of the document key:
  `sha256(doc_key) mod 1000`, validation `< 5`, test `< 10`.
- Dedup is set-based: the kept document of a duplicate cluster is the one
  with the smallest document key, independent of stream order.
- Shards and manifests are written atomically; every shard has a sha256.
- The manifest has a canonical digest over all fields except the digest.
- Tokenization order is sorted by document key; BPE training uses
  `num_threads=1`, `shuffle_input_sentence=False`, identity normalization.
- Sampling uses `numpy.random.Generator(PCG64(seed + epoch))`; the sequence
  permutation is recorded by hash.

HF streaming order is not guaranteed across library versions. The pinned
revision, the recorded source schema and the local materialized corpus are
the reproducibility anchor. A re-stream at the same revision is expected to
produce the same corpus; verification against the manifest decides.

## 5. Sources

Implemented in this step: FineWeb-Edu, config `sample-10BT`, public, ODC-By
1.0, revision pinned at download time. Two backends:

- `http`: paged rows from the Hugging Face datasets-server (100 rows per
  request, `revision` pinned). No extra dependency, bounded memory. This is
  the backend used for the pilot.
- `datasets`: `datasets.load_dataset(..., streaming=True)` when the library
  is installed in the project environment. The scale path; not installed in
  the system Python, so the pipeline does not require it.

Planned, documented extension points (not implemented): Wikimedia
Wikipedia, Cosmopedia v2, Python-Edu. Each needs its own schema, license and
access check before use.

## 6. Tokenizer

BPE, 8192 tokens, byte fallback, trained on the train split only. Special
tokens: `<unk>` 0, `<s>` 1, `</s>` 2, `<pad>` 3. Every document ends with
`</s>`; document offsets are stored next to the token array. Identity
normalization, no dummy prefix, no whitespace removal: the prepared text is
the tokenizer input, unchanged.

Model parameter effect for the v2 reference architecture: the untied token
embedding and output head grow by `(8192 - 1000) * 256 * 2 = 3,682,304`
parameters, so a v2-shaped model at 8192 tokens has about 29.28M total and
the same 9.89M active per token (embeddings are always active). The
vocabulary decision is recorded before any model work.

## 7. Budgets

| stage | budget | purpose |
|---|---|---|
| streaming smoke | 20 documents, one page | prove adapter, schema, revision |
| pilot | 2,000 documents, max 2M characters | prove the full pipeline and tokenizer |
| 10M-token pilot | about 10M tokens, about 40MB text | first training-data pilot, later |
| 1B-token corpus | later, multi-GB | real v3 pretraining, not this step |

The pilot root lives outside git (`--root`), with only manifests and reports
committed. No corpus or token shard is committed.

## 8. Milestones and acceptance

| milestone | acceptance |
|---|---|
| M0 frozen v2 | tag `v2-research-freeze-2026-10-26`, 89 tests green |
| M1 sources | schema, license, revision recorded; 20-doc streaming smoke passes; auth/network errors classified |
| M2 documents | deterministic shards, manifest digest reproducible, split isolation and dedup verified |
| M3 tokenizer + tokens | 8192-vocab BPE trained on train only; token shards reproducible; per-document decode matches |
| M4 loader | one epoch covers every sequence exactly once, no token reuse, packing boundaries tested |
| M5 pilot | 2,000-doc pilot materialized and verified; report with counts and hashes |

## 9. Risks and mitigations

| risk | mitigation |
|---|---|
| HF streaming order changes | pin revision, record schema, verify against the materialized manifest |
| datasets-server rate limits or caps | bounded pages, retry with backoff, document the `datasets` backend for scale |
| near-duplicate detection misses paraphrases | documented as near-identical detection (SimHash, Hamming <= 3), MinHash/embedding dedup is future work |
| memory growth of the dedup index | pilot scale now; bucketed LSH and external sorting before 10M documents |
| split leakage | split is a pure function of the doc key; duplicates are removed before assignment; tests check all three splits |
| tokenizer trained on leakage | training file contains train-split documents only, built from kept shards |
| license drift | license and revision in every manifest; DATA-SOURCES.md records the terms |
| accidental commits | pipeline data root is gitignored; no tokens or credentials are read or stored |

## 10. Next steps after this step

1. 10M-token pilot on the same pipeline with the `datasets` backend or the
   paged backend if rate limits allow.
2. Train the 8192-vocab tokenizer on the 10M-token train split and measure
   fertility on held-out text.
3. Only then define the v3 model width/depth and request GPU approval.
4. Compare against the frozen v2 numbers with the bits/char protocol.
