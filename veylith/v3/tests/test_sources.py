"""Tests for the source adapters: streaming, revision pinning, error classes."""

import json
import urllib.error

import pytest

from data_pipeline import sources


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def make_opener(items):
    calls = []

    def opener(request, timeout=None):
        calls.append({"url": request.full_url, "timeout": timeout,
                      "headers": dict(request.headers)})
        item = items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    return opener, calls


def rows_payload(count=1, features=("text", "id")):
    return {
        "features": [{"name": name} for name in features],
        "rows": [{"row_idx": index, "row": {"id": f"id-{index}",
                                            "text": "hello world " * 40}}
                 for index in range(count)],
        "num_rows_total": count,
        "partial": False,
    }


def test_http_source_pins_revision_and_retries_transient_errors():
    url = "https://datasets-server.huggingface.co/rows"
    items = [urllib.error.HTTPError(url, 500, "boom", {}, None), rows_payload(2)]
    opener, calls = make_opener(items)
    delays = []
    source = sources.FineWebEdu(revision="rev-123", opener=opener,
                                sleep=delays.append)
    documents = list(source.iter_documents(limit_rows=2))
    assert [doc.source_id for doc in documents] == ["id-0", "id-1"]
    assert documents[0].doc_key == "HuggingFaceFW/fineweb-edu@rev-123:id-0"
    assert "revision=rev-123" in calls[0]["url"]
    assert "offset=0" in calls[0]["url"]
    assert delays == [1.0]
    assert all("Authorization" not in call["headers"] for call in calls)


def test_http_auth_error_is_typed_and_mentions_no_credentials():
    url = "https://datasets-server.huggingface.co/rows"
    opener, calls = make_opener([urllib.error.HTTPError(url, 401, "denied", {}, None)])
    source = sources.FineWebEdu(revision="rev-123", opener=opener)
    with pytest.raises(sources.SourceAuthError) as err:
        list(source.iter_documents(limit_rows=1))
    assert "huggingface-cli login" in str(err.value)
    assert "hf_" not in str(err.value)
    assert len(calls) == 1


def test_http_network_error_after_bounded_retries():
    opener, calls = make_opener([urllib.error.URLError("offline")] * 3)
    delays = []
    source = sources.FineWebEdu(revision="rev-123", opener=opener,
                                sleep=delays.append)
    with pytest.raises(sources.SourceNetworkError):
        list(source.iter_documents(limit_rows=1))
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_http_schema_drift_is_rejected():
    opener, _ = make_opener([rows_payload(1, features=("text",))])
    source = sources.FineWebEdu(revision="rev-123", opener=opener)
    with pytest.raises(sources.SourceSchemaError) as err:
        list(source.iter_documents(limit_rows=1))
    assert "id" in str(err.value)


def test_http_404_is_a_source_error():
    url = "https://datasets-server.huggingface.co/rows"
    opener, _ = make_opener([urllib.error.HTTPError(url, 404, "missing", {}, None)])
    source = sources.FineWebEdu(revision="rev-123", opener=opener)
    with pytest.raises(sources.SourceError):
        list(source.iter_documents(limit_rows=1))


def test_local_source_offset_and_limit(tmp_path):
    path = tmp_path / "docs.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(json.dumps({"id": f"d{index}", "text": "text " * 50}) + "\n")
    source = sources.LocalJsonlSource(str(path), revision="local-rev")
    documents = list(source.iter_documents(offset=2, limit_rows=2))
    assert [doc.source_id for doc in documents] == ["d2", "d3"]
    assert documents[0].doc_key == f"local/jsonl@local-rev:d2"


def test_local_source_rejects_missing_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
    source = sources.LocalJsonlSource(str(path))
    with pytest.raises(sources.SourceSchemaError):
        list(source.iter_documents(limit_rows=1))


def test_planned_sources_are_documented_but_not_implemented():
    assert "wikipedia" in sources.PLANNED_SOURCES
    with pytest.raises(sources.SourceError) as err:
        sources.open_source("wikipedia")
    assert "extension point" in str(err.value)


def test_local_source_requires_a_path():
    with pytest.raises(sources.SourceError):
        sources.open_source("local-jsonl")
