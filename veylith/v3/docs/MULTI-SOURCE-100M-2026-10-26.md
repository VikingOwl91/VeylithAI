# Multi-source 100M-token corpus (2026-10-26)

Scope: veylith v3.0 pretraining data infrastructure, CPU only. One
multi-source pilot and one 100M-token corpus from the cleared sources. No
model training, no GPU, no architecture change, no claim about model quality.

Corpus root: `~/veylith-v3-data/multi-100m-20261026` (outside git).
Pilot root: `~/veylith-v3-data/multi-pilot-20261026` (outside git).
Metadata committed: `corpus-100m-manifest-2026-10-26.json`,
`multi-source-report-100m-2026-10-26.json`,
`loader-benchmark-100m-2026-10-26.json`.

## Summary

| item | value |
|---|---|
| sources planned | FineWeb-Edu 60 %, Wikipedia 20 %, Cosmopedia v2 15 %, Python-Edu 5 % |
| sources used | FineWeb-Edu, Wikipedia, Cosmopedia v2 |
| blocked source | Python-Edu: no text in the HF dataset, text requires The Stack v2 (gated, license `other`); terms not accepted -> blocked |
| rows read | 97,280 |
| accepted / available | 95,188 / 95,180 |
| selected documents | 90,784 (train 87,063, validation 1,874, test 1,847) |
| train tokens | 95,951,233 |
| validation / test tokens | 2,122,377 / 2,094,292 |
| total tokens | 100,167,902 |
| tokenizer | BPE 8192, sha256 `411227a8e5289d6f4b450059b744ed278532a4624d3895f1a4979600c49ae034` |
| manifest digest | `7a6e7991dd83244ced4c9c13627028e1824f00b0f58735cc665323ff111bdcf0` |
| root size | 1.9 GB (cap 5 GiB) |
| prepare / tokenize / verify | 811 s / 98 s / 375 s |
| peak RSS | 2.78 GB prepare, 3.68 GB tokenize, 0.91 GB verify |
| loader | 374,810 sequences, 1.10M tokens/s, 227 MB peak, resume equal |
| verification | deep + near-duplicate check green |

The train target of 100M is missed by 4.05M tokens because Python-Edu is
blocked. The three cleared sources hit their quotas within 1.7 %. No source
was increased to compensate; the gap is reported, not filled.

## 1. Sources checked

| source | dataset id / config | revision | license | rows | schema | status |
|---|---|---|---|---:|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` / `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0 | 9,672,101 | text, id, dump, url, file_path, language, language_score, token_count, score, int_score | used |
| Wikipedia | `wikimedia/wikipedia` / `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC BY-SA 3.0 + GFDL | 6,407,814 | id, url, title, text | used |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` / `cosmopedia-v2` | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-By 1.0 (corpus card) | 39,134,000 | prompt, text, token_length, audience, format, seed_data | used |
| Python-Edu | `HuggingFaceTB/smollm-corpus` / `python-edu` | `3ba9d605774198c5868892d7a8deda78031a781f` | `other` (The Stack v2) | 7,678,448 metadata rows | blob_id, repo_name, path, length_bytes, score, int_score | **blocked** |

Python-Edu has no text column. The SmolLM card says the file contents are
downloaded from the Software Heritage S3 bucket and refers to The Stack v2
for the license. The Stack v2 is gated (`license: other`) and its card is not
accessible without accepting its terms (HTTP 401). No terms were accepted on
the user's behalf; the source is blocked and the 5 % share is missing.

## 2. Pilot (about 1M train tokens per cleared source)

| source | docs (train/val/test) | train tokens | heldout chars/token | fertility |
|---|---|---:|---:|---:|
| FineWeb-Edu | 1,079 / 14 / 7 | 1,019,933 | 3.9723 | 1.6096 |
| Wikipedia | 159 / 0 / 0 | 1,243,821 | not measurable (0 heldout) | not measurable |
| Cosmopedia v2 | 1,115 / 8 / 9 | 904,225 | 4.4489 | 1.6017 |
| total | 2,231 / 22 / 16 | 3,167,979 | - | - |

Pilot wall 40.1 s, peak 728 MB; tokenizer `f5001345...`; deep + near-dup
verification green; loader 12,375 sequences, 16.8M tokens/s, 47 MB, resume
equal.

The pilot exposed one issue: at 159 Wikipedia documents the 1 %
hash split produced zero validation and test documents. The 100M plan uses
2 % validation and 2 % test (20/20 permille), which gives 167 and 152
Wikipedia documents and 0.51M / 0.47M Wikipedia tokens per held-out split.

## 3. Final corpus distribution

| source | train docs | train chars | train tokens | share of train | quota | deviation |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb-Edu | 62,341 | 238,338,944 | 60,356,799 | 62.90 % | 60,000,000 | +0.59 % |
| Wikipedia | 6,873 | 72,609,738 | 20,335,474 | 21.19 % | 20,000,000 | +1.68 % |
| Cosmopedia v2 | 17,849 | 66,737,167 | 15,258,960 | 15.90 % | 15,000,000 | +1.73 % |
| Python-Edu | 0 | 0 | 0 | 0 % | 5,000,000 | blocked |
| total | 87,063 | 377,685,849 | 95,951,233 | 100 % | 100,000,000 | -4.05 % |

The quota targets were 60/20/15 for the cleared sources and 5 for Python-Edu.
The actual token counts come from the shared 8192 tokenizer; the per-source
character-per-token estimates from the pilot were accurate enough to land
within 1.7 %. Validation and test are outside the quota: 1,874 and 1,847
documents, 2.12M and 2.09M tokens, 2.06 % and 2.03 % of the selected corpus.

Caps: 95,951,233 train tokens (< 150M), 100,167,902 total tokens, 1.9 GB (<
5 GiB), 97,280 source rows (< 1M), 811 s network/prepare (< 3 h).

## 4. Tokenizer

BPE 8192, trained on the merged train split only (87,063 documents, 377.9M
characters). Model `411227a8...`, vocab `d5c6e6cb...`, training text
`1f2ffe5b...`. Per-source held-out fertility:

| source | heldout docs | chars/token | fertility (tokens/word) |
|---|---:|---:|---:|
| FineWeb-Edu | 2,692 | 3.9458 | 1.5533 |
| Wikipedia | 319 | 3.6082 | 1.7487 |
| Cosmopedia v2 | 710 | 4.3743 | 1.5668 |

Special tokens `<unk>` 0, `<s>` 1, `</s>` 2, `<pad>` 3. Wikipedia and code-like
content tokenize less efficiently than web prose; Cosmopedia is the most
efficient because its synthetic prose is regular. Validation and test were
not used for tokenizer training.

## 5. Deduplication and split isolation

Documented order: per-source streaming/filtering -> within-source exact/near
dedup -> cross-source exact dedup -> cross-source near dedup -> deterministic
split assignment -> per-source quota filling on train -> deterministic bucket
shuffle for shard writing. Quotas are applied after dedup and splits, so
duplicates never consume budget; a source that cannot fill its quota is not
compensated by others.

- Rejected: 1,990 `too_long`, 96 `too_short`, 3 `letter_ratio`, 1
  `replacement_chars`.
- Within-source duplicates removed before the merge: 5 exact, 3 near.
- Cross-source exact duplicates: 0. Cross-source near duplicates: 0.
- Post-hoc check over all 90,784 selected documents: 0 pairs at Hamming <= 3,
  0 skipped LSH buckets.
- Split isolation: every document is in exactly one split; `verify_corpus`
  recomputes the split for every doc_key and the per-source counts.
- No near-duplicate crosses a split.

The zero cross-source count is a property of this sample (different sources,
synthetic text), not a proof that the sources never overlap. The dedup
mechanism itself is tested with synthetic cross-source duplicates.

## 6. Shuffle, shards and provenance

- Deterministic quota order: `sha256(seed:doc_key)` with seed 0; per-source
  train documents are taken in that order until the char budget is reached.
- Deterministic shard shuffle: 64 buckets from
  `int(sha256(seed:doc_key)[:8],16) % 64`, buckets written in order. Records
  carry `source` and source metadata.
- The manifest lists every source with dataset id, config, revision, license,
  access, attribution and provenance; per-source document/char/token counts;
  quotas and shortfalls; 175 train, 4 validation and 4 test shards with
  sha256.
- The manifest's `source` field is a compatibility field (first sorted
  source); the authoritative source list is `sources` and
  `counts.by_source`.

## 7. Runtime, RAM and storage

| phase | wall | peak RSS |
|---|---:|---:|
| multi-source prepare + merge (100M) | 811 s | 2.78 GB |
| tokenize 8192 | 98 s | 3.68 GB |
| verify deep + near-dups | 375 s | 908 MB |
| loader plan + 200 batches | 0.84 s + 0.37 s | 227 MB |

Root 1.9 GB: staging ~800 MB, per-source corpora ~800 MB, merged documents
~390 MB, token shards ~200 MB, tokenizer train text ~378 MB. The HF
`datasets` cache is outside the root and not counted in the 5 GiB cap;
it stays below 1 GB for this pilot. The 2.78 GB prepare peak is dominated by
the in-memory metadata and the merge bucket pass; 18 GB were available.

## 8. Reproducibility

- All sources use pinned revisions; every manifest records them.
- Splits are a pure function of `doc_key`; quota selection is a pure function
  of `(seed, doc_key, chars)`.
- The pilot and the 100M corpus were produced with the same code and plan
  format; the pilot validated adapters, dedup, tokenization and loader before
  the larger run.
- Tokenizer model sha256, vocab sha256, train-text sha256 and all shard
  sha256 are in the committed manifest.
- The loader order digest is a pure function of `(seed, epoch, plan)`;
  `benchmark_loader.py` checks mid-epoch resume equality.
- Tests: v3 43 passed + 2 skipped in the project environment (44 + 1 on the
  system Python, where the optional `datasets` backend is absent); v2 89
  passed. Network smoke and the missing-dependency path are covered.
- A second full 100M network materialization was not attempted; the pinned
  revisions, the deterministic plan and the committed manifest are the
  anchors. The earlier 10M pilot documented the HTTP 429 behavior.

## 9. Risks and limitations

- **Python-Edu blocked**: 5 % of the target mix is missing; no substitute was
  used. The Stack v2 terms must be reviewed and accepted by the user before
  this source can be added.
- **Wikipedia license**: CC BY-SA 3.0 and GFDL. Local research use is
  documented; redistribution of the corpus would carry attribution and
  share-alike obligations.
- **Synthetic data**: Cosmopedia v2 is Mixtral-generated from web seeds. It
  can contain hallucinations, stale facts and model-specific style. It is
  15.9 % of the train tokens.
- **Contamination**: Cosmopedia seeds come from FineWeb; near-duplicate
  dedup removes near-identical text but not semantic overlap. The
  evaluation split is document-hash based, not source-aware; per-source
  evaluation is reported.
- **PII/copyright**: FineWeb is filtered web text, Wikipedia is public
  encyclopedia text, Cosmopedia is synthetic. No additional PII scrubbing
  beyond source filters and the length/quality filters. Copyright status of
  individual web pages is not verified; the local corpus is not published.
- **Wikipedia coverage**: 6,873 train documents with long articles; 167/152
  held-out documents. Token coverage is good, document diversity is limited.
- **Tokenizer**: 8192 BPE, no vocabulary ablation on the multi-source corpus;
  per-source fertility differs (Wikipedia 1.75 vs 1.55 for the others).
- **No model quality claim**: the corpus is not trained on in this task.

## 10. Recommendation

1. Use this 95.95M-token corpus as the v3.1 data baseline; keep the 5M gap
   documented and do not backfill it silently.
2. For v3.1, see `V3.1-DESIGN.md`: smaller MoE (17.3M total / 8.43M active
   at 8192) as primary, dense 8.3M baseline as compute-matched control, the
   25.6M v2 MoE as upper-capacity reference.
3. Evaluate per source with bits/char and main CE on the frozen validation
   and test splits; report the synthetic share and the Wikipedia share
   separately.
4. If Python-Edu is needed, first obtain and document The Stack v2 terms,
   then implement the S3 fetch path and add the source to a new plan; do not
   silently substitute.
5. No 100M+ escalation without a new budget and approval.
