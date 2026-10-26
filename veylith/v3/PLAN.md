# veylith v3: data pipeline plan

Status 2026-10-26. Branch `v3/data-pipeline`, based on the frozen v2 tag
`v2-research-freeze-2026-10-26`. This document is the plan for the first v3
step: a modular, streaming-capable pretraining data pipeline. It does not
change the v2 baseline.

## Project classification

v1.x and v2.x are experimental research prototypes and feasibility studies,
not finished or production-ready language models. The version labels are:

| label | meaning |
|---|---|
| v1.x | Experimental Transformer Baselines |
| v2.x | Experimental MoE Architecture Research |
| v3.0 | Pretraining Data Infrastructure |
| v3.1 | Initial Pretraining Experiments |

This plan covers v3.0 only. It produces data, not a model. v3.1 starts only
after the data pipeline and tokenizer are validated and a GPU run is
explicitly approved.

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
- No 100M-token download and no model training in this step.
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
13.57M active per token under the audit convention (the 9.89M of the 1000
vocabulary plus the 3.68M embedding/head delta; embeddings count as active).
The vocabulary decision is recorded before any model work.

## 7. Budgets

| stage | budget | purpose |
|---|---|---|
| streaming smoke | 100 documents, one page | prove adapter, schema, revision |
| pilot | 2,000 documents, max 2M characters | prove the full pipeline and tokenizer |
| 10M-token pilot | 11,400 documents, 10,333,164 train tokens, 194 MB | done 2026-10-26; 8192 BPE |
| 100M multi-source corpus | 90,784 documents, 95,951,233 train tokens, 1.9 GB | done 2026-10-26; FineWeb/Wikipedia/Cosmopedia, Python-Edu blocked |
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

## 10. Milestone status

| milestone | status |
|---|---|
| M0 frozen v2 | done, tag `v2-research-freeze-2026-10-26` |
| M1 sources | done, FineWeb-Edu smoke at a pinned revision |
| M2 documents | done, deterministic shards, dedup and split tests |
| M3 tokenizer and tokens | done, 8192 BPE on train only, deep verification |
| M4 loader | done, concat and document packing, deterministic sampling |
| M5 pilot | done, 2000 documents / 1.92M tokens, verified |
| 10M-token pilot | done, 11,399 documents / 10.33M train tokens, 8192 BPE, deep + near-dup verification green |
| 100M multi-source corpus | done, 90,784 documents / 95.95M train tokens, 8192 BPE, deep + near-dup verification green; Python-Edu blocked |
| v3.1 design note | done, `docs/V3.1-DESIGN.md`; no model code or GPU run |

## 11. Next steps after this step

1. v3.1 starts only with a separate approval: choose the candidate from
   `docs/V3.1-DESIGN.md` (smaller MoE primary, dense baseline control,
   v2 MoE reference), fix the command, then exactly one GPU run.
2. Compare against the frozen v2 numbers with the bits/char protocol; report
   per source and never compare raw token losses across tokenizers.
3. Use the 8192 tokenizer as the preliminary default; per-source fertility
   is 1.55 (FineWeb), 1.75 (Wikipedia), 1.57 (Cosmopedia).
4. Python-Edu needs The Stack v2 terms review and acceptance before it can
   be added; do not substitute or backfill the 5 % gap silently.
5. No 100M+ escalation or automatic follow-up run.
