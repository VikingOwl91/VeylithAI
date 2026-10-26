"""Shared setup for the v3 data pipeline tests (CPU only, no network)."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
for path in (V3, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import support  # noqa: E402


@pytest.fixture
def source_file(tmp_path):
    docs = support.make_docs()
    path = tmp_path / "source.jsonl"
    support.write_source(path, docs)
    return path, docs


@pytest.fixture
def fake_source():
    return support.FakeSource(support.make_docs())
