"""veylith v3 pretraining data pipeline.

CPU-only, dependency-light (stdlib, numpy, sentencepiece). The components are
independent:

    sources  -> dataset adapters with bounded streaming
    prepare  -> normalization, filtering, split assignment, deduplication
    dataset  -> document shards, manifests, atomic writes, verification
    tokenize -> BPE training on the train split, token shards with doc offsets
    loader   -> deterministic sequence packing and batch sampling
"""

PIPELINE_VERSION = "0.1.0"
FILTER_VERSION = "v1"
FORMAT_VERSION = 1
