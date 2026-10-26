# Reproducing the v3 data pipeline

All commands are CPU-only and run from the repository root. No GPU, no
system package, no network for the test suite.

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
python3 -m pytest veylith/v3/tests -q
```

Expected offline result: 30 passed, 1 skipped (the network smoke is skipped
unless `VEYLITH_NETWORK_TESTS=1`). The suite uses a fake source and a local
JSONL source; it writes only to pytest temporary directories.

The full repository suite:

```bash
python3 -m pytest veylith/v2/tests veylith/tests veylith/v3/tests -q
```

## Streaming smoke (network)

```bash
python3 veylith/v3/pipeline.py inspect --source fineweb-edu --limit 20
```

This resolves the revision from the Hub, requests one page of 20 documents
from the datasets-server and writes nothing. On 2026-10-26 it returned 20
documents at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, schema
`text, id, dump, url, file_path, language, language_score, token_count,
score, int_score`.

## Pilot corpus (2000 documents)

The pilot lives outside the repository:

```bash
ROOT=~/veylith-v3-data/pilot-20261026

python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 2000 \
    --root "$ROOT" --shard-docs 500 --min-chars 200 --max-chars 20000 \
    --val-permille 5 --test-permille 5

python3 veylith/v3/pipeline.py tokenize --root "$ROOT" --vocab 8192
python3 veylith/v3/pipeline.py verify --root "$ROOT" --deep
python3 veylith/v3/pipeline.py batches --root "$ROOT" --seq-len 256 \
    --batch-size 4 --batches 2
```

Reference values of the pilot materialized on 2026-10-26:

| item | value |
|---|---|
| rows read | 2068 |
| documents kept | 2000 (train 1986, validation 5, test 9) |
| rejected | 68 `too_long` |
| duplicates | 0 exact, 0 near (in this sample) |
| characters | 7,469,712 train, 14,387 validation, 28,168 test |
| tokens | 1,906,893 train, 3,737 validation, 7,499 test (1,918,129 total) |
| chars per token | about 3.92 |
| manifest digest | `16af7cf3f93a598ff5fea898c0b4fd3323a0dbdec3ad94b386dedb81e6c380c4` |
| tokenizer model sha256 | `dd738886aea9e9ec27b114ecb75d152b1890a87d88e24e2664cba04d755c8663` |
| tokenizer vocab sha256 | `bb167f10025eb6e2b5360d116f5ef06f4ce9e0ee798ba25dcb77c84abaf8cd0d` |
| train text sha256 | `3f9fbdf855f315ceb533ab2971631fe657fc08a8530d0f62e94723568876dc73` |
| source revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| source license | ODC-By 1.0 |
| disk | about 29 MB (staging plus corpus plus tokens) |

The digest depends on the exact source rows, filters, split permille values
and shard sizes. A later run at the same revision and settings should
reproduce it; a changed revision, page composition or filter will not.

The full manifest of this pilot (source revision, license, counts, every
shard sha256, tokenizer hashes) is committed at
`veylith/v3/docs/pilot-manifest-2026-10-26.json`. It contains no source
text. The corpus itself stays outside git.

`verify --deep` decodes every token span, strips EOS and compares the text
hash with the document shard. It also checks every shard sha256, the
manifest digest, split isolation and the split policy.

## Resume

`prepare` writes staging shards and `staging/state.json` at every flush.
Resume continues from the recorded source offset and skips documents
already in the staging index:

```bash
python3 veylith/v3/pipeline.py prepare --source fineweb-edu --docs 4000 \
    --root "$ROOT" --resume
```

A resume is refused when the source identity (dataset, config, split,
revision), the filter configuration or the pipeline version changed. The
offline test `test_resume_matches_an_uninterrupted_run` proves that a
resumed run produces the same manifest digest and shard hashes as a single
uninterrupted run.

The download phase re-streams from the recorded offset. The paged backend
does not seek server-side; it starts at `offset` in the page query, so the
source rows before the offset are not fetched. The `datasets` backend uses
`.skip(offset)`, which reads through preceding rows without yielding them.

## Determinism notes

- Splits, dedup decisions and document ordering do not depend on arrival
  order. Exact and near duplicates keep the smallest `doc_key`.
- The tokenizer model is byte-identical across roots: training uses
  `sentence_iterator` plus `model_writer`, so no root-specific path is
  embedded. The offline test trains twice and compares the model sha256.
- `batches` uses `SeedSequence([seed, epoch])` and returns the permutation
  digest with every batch; one epoch covers every sequence exactly once.
- The HF stream itself is not a stable contract. The pinned revision and the
  local manifest are the anchors. A second full 2000-document materialization
  on 2026-10-26 hit the datasets-server rate limit (HTTP 429 after three
  retries); this is a backend limit, not a pipeline failure. Use the
  `datasets` backend or wait between large runs.

## Recreate or remove

The pilot is disposable and not in git:

```bash
rm -rf ~/veylith-v3-data/pilot-20261026   # delete
# rerun the three commands above to recreate
```

No credential, corpus shard or tokenizer is stored in the repository.
