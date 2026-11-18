# Veylith training data

## v1, v2 and Phase 4 · Tiny Shakespeare

William Shakespeare wrote the plays. Andrej Karpathy distributed the Tiny Shakespeare compilation in the 2015 *char-rnn* project. Both trainers load this corpus, and the Phase 4 batch-delivery experiment used token IDs derived from it. The repository includes the text twice, in `v1/data/tinyshakespeare.txt` and `v2/data/tinyshakespeare.txt`, alongside tokenizer files and token-ID caches. The selection and evaluation offsets refer to the same text. [v1 trainer](v1/train.py) · [v2 trainer](v2/train.py) · [Phase 4 run](debug/batch-delivery.md)

Both tracked texts match the upstream file at commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3` byte for byte by SHA-256, `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. This identifies the content even though the training scripts use a mutable download URL. [Pinned upstream text](https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt)

Shakespeare's original plays are public domain. The char-rnn README labels the project MIT. Karpathy's dataset card leaves licensing information unspecified. The precise edition and any separate grant covering editorial work on this compilation remain unconfirmed. Credit Shakespeare and Karpathy when reusing the supplied text. [Upstream project notice](https://github.com/karpathy/char-rnn#license) · [Dataset card](https://huggingface.co/datasets/karpathy/tiny_shakespeare#licensing-information)

## v3 · Pretraining corpora

The early tokenizer and data-pipeline pilots used FineWeb-Edu. The multi-source corpus adds English Wikipedia and Cosmopedia v2. The v3.1 dense training record identifies this corpus by its manifest digest and records 95,951,233 available train-split tokens. The count describes corpus size. Individual runs consumed the amounts in their run records. Veylith distributes manifests and measurements for these corpora. [Source inventory](v3/docs/DATA-SOURCES.md) · [100M manifest](v3/docs/corpus-100m-manifest-2026-10-26.json) · [Dense training record](v3/docs/dense-run-v31l-2026-10-27-summary.json)

| Source and credit | Origin and dataset terms |
|---|---|
| FineWeb-Edu · Hugging Face FineWeb team | Educational subset of FineWeb, derived from Common Crawl web pages. Veylith used the English `sample-10BT` configuration. The database is available under ODC-By 1.0. Attribute FineWeb-Edu and preserve the source notices. Rights in individual web pages can differ. [Pinned dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md) · [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| English Wikipedia · Wikipedia contributors, distributed by Wikimedia | English dump dated 2023-11-01, configuration `20231101.en`. The dataset card lists CC BY-SA 3.0 and GFDL and documents article-level exceptions. When redistributing text, retain article attribution and source links, identify adaptations, and follow the applicable share-alike and notice terms. [Pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| Cosmopedia v2 · Hugging Face SmolLM team | Synthetic educational text generated with Mixtral-8x7B-Instruct-v0.1 using web-derived prompts. Veylith used the `cosmopedia-v2` configuration of SmolLM-Corpus. The publisher lists ODC-By 1.0 for this corpus. Credit the dataset and retain its notices. [Pinned dataset card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md) · [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) |

**Python-Edu contributed zero training text.** The planned component exposed metadata pointing to files in Software Heritage. Retrieval under The Stack v2 terms was blocked, and its planned share was left unfilled. [Materialization report](v3/docs/MULTI-SOURCE-100M-2026-10-26.md)
