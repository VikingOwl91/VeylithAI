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
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener,
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
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener)
    with pytest.raises(sources.SourceAuthError) as err:
        list(source.iter_documents(limit_rows=1))
    assert "huggingface-cli login" in str(err.value)
    assert "hf_" not in str(err.value)
    assert len(calls) == 1


def test_http_network_error_after_bounded_retries():
    opener, calls = make_opener([urllib.error.URLError("offline")] * 3)
    delays = []
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener,
                                sleep=delays.append)
    with pytest.raises(sources.SourceNetworkError):
        list(source.iter_documents(limit_rows=1))
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_http_schema_drift_is_rejected():
    opener, _ = make_opener([rows_payload(1, features=("text",))])
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener)
    with pytest.raises(sources.SourceSchemaError) as err:
        list(source.iter_documents(limit_rows=1))
    assert "id" in str(err.value)


def test_http_404_is_a_source_error():
    url = "https://datasets-server.huggingface.co/rows"
    opener, _ = make_opener([urllib.error.HTTPError(url, 404, "missing", {}, None)])
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener)
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


def test_python_edu_is_blocked_by_license_and_access():
    assert "python-edu" in sources.BLOCKED_SOURCES
    with pytest.raises(sources.SourceBlockedError) as err:
        sources.open_source("python-edu")
    assert "Stack v2" in str(err.value)


def test_hf_datasets_source_uses_id_fields(monkeypatch):
    import sys
    import types

    rows = [{"id": "7", "text": "Anarchism " * 50, "title": "Anarchism",
             "url": "https://en.wikipedia.org/wiki/Anarchism"}]

    def fake_load(dataset, name=None, split=None, streaming=False, revision=None):
        assert streaming is True
        assert revision == "rev-1"
        return iter(rows)

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=fake_load))
    wikipedia = sources.HFDatasetsSource("wikipedia", revision="rev-1")
    documents = list(wikipedia.iter_documents(limit_rows=1))
    assert documents[0].doc_key == "wikimedia/wikipedia@rev-1:7"
    assert documents[0].meta["title"] == "Anarchism"
    assert documents[0].meta["url"].startswith("https://en.wikipedia.org/")


def test_hf_datasets_source_uses_content_hash_for_cosmopedia(monkeypatch):
    import hashlib
    import sys
    import types

    text = "synthetic textbook " * 60
    rows = [{"text": text, "audience": "college_students", "format": "textbook",
             "seed_data": "fineweb", "token_length": 805}]

    def fake_load(dataset, name=None, split=None, streaming=False, revision=None):
        assert dataset == "HuggingFaceTB/smollm-corpus"
        assert name == "cosmopedia-v2"
        assert revision == "rev-1"
        return iter(rows)

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=fake_load))
    cosmopedia = sources.HFDatasetsSource("cosmopedia-v2", revision="rev-1")
    documents = list(cosmopedia.iter_documents(limit_rows=1))
    expected = hashlib.sha256(text.encode()).hexdigest()[:32]
    assert documents[0].doc_key == f"HuggingFaceTB/smollm-corpus@rev-1:{expected}"
    assert documents[0].meta["seed_data"] == "fineweb"


def test_hf_datasets_source_rejects_a_missing_text_field(monkeypatch):
    import sys
    import types

    def fake_load(dataset, name=None, split=None, streaming=False, revision=None):
        return iter([{"id": "1"}])

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=fake_load))
    source = sources.HFDatasetsSource("wikipedia", revision="rev-1")
    with pytest.raises(sources.SourceSchemaError):
        list(source.iter_documents(limit_rows=1))


def test_hf_datasets_source_classifies_auth_errors(monkeypatch):
    import sys
    import types

    def fake_load(dataset, name=None, split=None, streaming=False, revision=None):
        raise OSError("401 Client Error: Unauthorized")

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=fake_load))
    source = sources.HFDatasetsSource("wikipedia", revision="rev-1")
    with pytest.raises(sources.SourceAuthError):
        list(source.iter_documents(limit_rows=1))


def test_local_source_requires_a_path():
    with pytest.raises(sources.SourceError):
        sources.open_source("local-jsonl")


def test_http_429_uses_a_longer_bounded_backoff():
    url = "https://datasets-server.huggingface.co/rows"
    items = [urllib.error.HTTPError(url, 429, "rate limited", {}, None),
             rows_payload(1)]
    opener, calls = make_opener(items)
    delays = []
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener,
                                sleep=delays.append)
    documents = list(source.iter_documents(limit_rows=1))
    assert len(documents) == 1
    assert delays == [10.0]


def test_page_delay_is_applied_only_between_pages():
    opener, calls = make_opener([rows_payload(100), rows_payload(1)])
    delays = []
    source = sources.FineWebEdu(revision="rev-123", backend="http", opener=opener,
                                sleep=delays.append, page_delay=0.25)
    documents = list(source.iter_documents(limit_rows=101))
    assert len(documents) == 101
    assert delays == [0.25]
    assert len(calls) == 2


def test_datasets_backend_reports_a_missing_dependency():
    import importlib.util
    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets is installed; the missing-dependency path is not reachable")
    source = sources.FineWebEdu(revision="rev-123", backend="datasets")
    with pytest.raises(sources.MissingDependencyError) as err:
        list(source._iter_datasets("rev-123", 0, None, 1))
    assert "not installed" in str(err.value)
