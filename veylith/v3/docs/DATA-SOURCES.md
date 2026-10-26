# v3 data sources and licenses

Checked on 2026-10-26 against the Hugging Face API and the datasets-server.
The repository does not redistribute any source text; the pilot corpus stays
outside git.

## Implemented: FineWeb-Edu

| field | value |
|---|---|
| dataset id | `HuggingFaceFW/fineweb-edu` |
| config | `sample-10BT` |
| split | `train` |
| revision (pinned) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| license | ODC-By 1.0 (`odc-by` on the dataset card) |
| access | public, `gated: false`, `private: false` |
| rows (datasets-server) | 9,672,101 for `sample-10BT` |
| text field | `text` (English, educational web text) |
| id field | `id` (stable `urn:uuid:...`) |
| metadata fields | `dump`, `url`, `file_path`, `language`, `language_score`, `token_count`, `score`, `int_score` |

The dataset card describes FineWeb-Edu as a 1.3T-token educational subset of
FineWeb, filtered by an educational-quality classifier; `sample-10BT` is the
10B-token sample. It is web text, not curated prose, and the quality filter
is not a safety or factuality filter.

License obligations: ODC-By requires attribution. If a derived database is
published, attribute the FineWeb/FineWeb-Edu dataset and keep the license
notice. The manifest of every corpus records the license id; the local
pilot is not published.

Attribution used in this project:

> This work uses the FineWeb-Edu dataset (Hugging Face), licensed under the
> Open Data Commons Attribution License (ODC-By) v1.0.

## Access and credentials

Public sources need no token. For a gated dataset, log in with Hugging
Face's own tooling (`huggingface-cli login`); the credential stays in Hugging
Face's store. The pipeline never reads `HF_TOKEN`, never writes it to a
manifest or log, and never prints request headers. HTTP 401/403 is reported
as an access error with a pointer to that login, exit code 2.

## Backends

| backend | mechanism | dependency | use |
|---|---|---|---|
| `http` (default) | datasets-server `/rows`, 100 rows per request, `revision` pinned, bounded retries | none | pilot, constrained environments |
| `datasets` | `datasets.load_dataset(..., streaming=True)` | `datasets` (not installed in the system Python) | scale path |

The rows API is paginated and rate limited. The pilot's 21 requests for
2068 rows succeeded; an immediate second full materialization hit HTTP 429
after three bounded retries (exit 3, no partial manifest). Long runs should
use the `datasets` backend or space requests out. HF streaming order is not
a stable contract across library versions; the pinned revision plus the
local manifest is the reproducibility anchor.

## Planned sources (documented extension points)

Each entry must be checked for schema, revision, license and access before
implementation; their absence here is intentional.

| name | dataset id | config | status | license to verify |
|---|---|---|---|---|
| wikipedia | `wikimedia/wikipedia` | `20231101.en` | not implemented | CC BY-SA 4.0 (check card) |
| cosmopedia-v2 | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` | not implemented | check card before use |
| python-edu | `HuggingFaceTB/smollm-corpus` | `python-edu` | not implemented | check card before use |

Requirements for adding a source:

1. Record the dataset id, config, split, revision sha, license and access
   mode in `sources.py` and the manifest.
2. Implement a streaming adapter that yields `SourceDocument` with a stable
   `doc_key`; never read a token.
3. Check the schema against a bounded sample; schema drift must raise a
   typed error.
4. Add a license note to this file and an offline test with a fake source.
5. For share-alike sources (for example Wikipedia), decide explicitly how
   the derived corpus is licensed before redistribution.

## Corpus handling

- Normalization: NFC, CRLF to LF, control characters removed, trailing
  spaces stripped, 3+ blank lines collapsed. Content is otherwise unchanged.
- Filters: minimum and maximum length, replacement-character ratio, letter
  ratio, language and language score, `int_score >= 3` when present.
- Deduplication: exact and near-identical, global over the staged set; the
  kept document is the one with the smallest `doc_key`.
- Splits: pure function of `doc_key`, test < 5 permille, validation < 10
  permille, train otherwise. Change the permille values per run and record
  them in the manifest; never assign splits by arrival order.
- The tokenizer is trained on final train-split documents only.
