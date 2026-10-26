"""Dataset adapters with bounded, revision-pinned streaming.

Two backends for FineWeb-Edu:

* ``http`` (default when ``datasets`` is absent): paged rows from the Hugging
  Face datasets-server. 100 rows per request, bounded memory, no source
  download, no extra dependency.
* ``datasets``: ``datasets.load_dataset(..., streaming=True)``. The scale path;
  requires the library in the project environment.

The adapters never read, store or log authentication tokens. Public datasets
need none; for gated datasets the user logs in with Hugging Face's own tooling
and the ambient credential store is used by the HTTP stack.

HF streaming order is not a stable contract. The source revision is pinned and
the local materialized corpus is the reproducibility anchor (see PLAN.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

HUB_API = "https://huggingface.co/api/datasets/{dataset}"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
USER_AGENT = "veylith-v3-data-pipeline/0.1"


class SourceError(RuntimeError):
    """Base class for acquisition errors."""


class SourceAuthError(SourceError):
    """Access denied. No credential is stored or printed by the pipeline."""


class SourceNetworkError(SourceError):
    """Transient or permanent network failure after retries."""


class SourceSchemaError(SourceError):
    """The dataset does not expose the expected fields."""


class MissingDependencyError(SourceError):
    """An optional backend is not installed."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: str
    config: str
    split: str
    text_field: str = "text"
    id_field: str = "id"
    license_name: str = ""
    license_url: str = ""
    access: str = "public"
    notes: str = ""


SOURCES = {
    "fineweb-edu": SourceSpec(
        name="fineweb-edu",
        dataset_id="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        id_field="id",
        license_name="odc-by",
        license_url="https://opendatacommons.org/licenses/by/1-0/",
        access="public",
        notes="educational web text, FineWeb-Edu classifier score >= 3",
    ),
    "local-jsonl": SourceSpec(
        name="local-jsonl",
        dataset_id="local/jsonl",
        config="-",
        split="train",
        text_field="text",
        id_field="id",
        license_name="unknown",
        license_url="",
        access="local",
        notes="offline adapter for tests and user-provided corpora",
    ),
}

# Documented extension points. Each needs its own schema, license and access
# check before it can be used.
PLANNED_SOURCES = {
    "wikipedia": {"dataset_id": "wikimedia/wikipedia", "config": "20231101.en",
                  "license": "CC BY-SA 4.0"},
    "cosmopedia-v2": {"dataset_id": "HuggingFaceTB/smollm-corpus",
                      "config": "cosmopedia-v2", "license": "see dataset card"},
    "python-edu": {"dataset_id": "HuggingFaceTB/smollm-corpus",
                   "config": "python-edu", "license": "see dataset card"},
}


@dataclass
class SourceDocument:
    doc_key: str
    source: str
    dataset_id: str
    revision: str
    source_id: str
    text: str
    meta: dict = field(default_factory=dict)
    arrival_index: int = -1


# ------------------------------------------------------------------ HTTP

def _http_json(url: str, *, timeout: float = 60.0, retries: int = 3,
               sleep=time.sleep, opener=urllib.request.urlopen):
    """GET JSON with bounded retries. Raises typed SourceError subclasses."""
    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with opener(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SourceAuthError(
                    f"access denied (HTTP {exc.code}) for {url.split('?')[0]}; "
                    f"the pipeline stores no credentials - log in with "
                    f"'huggingface-cli login' if the dataset is gated") from None
            if exc.code == 404:
                raise SourceError(f"not found (HTTP 404): {url.split('?')[0]}") from None
            if exc.code == 429 or 500 <= exc.code < 600:
                last_error = exc
                if attempt + 1 < retries:
                    delay = 1.0 * (2 ** attempt)
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after and str(retry_after).isdigit():
                        delay = min(float(retry_after), 30.0)
                    sleep(delay)
                    continue
                raise SourceNetworkError(
                    f"HTTP {exc.code} after {retries} attempts: {url.split('?')[0]}") from None
            raise SourceError(f"HTTP {exc.code}: {url.split('?')[0]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep(1.0 * (2 ** attempt))
                continue
            raise SourceNetworkError(
                f"network error after {retries} attempts: {type(exc).__name__}") from None
        except json.JSONDecodeError as exc:
            raise SourceError(f"invalid JSON from {url.split('?')[0]}: {exc}") from None
    raise SourceNetworkError(f"request failed: {last_error}")


def fetch_dataset_info(dataset_id: str, *, timeout: float = 60.0,
                       retries: int = 3, sleep=time.sleep,
                       opener=urllib.request.urlopen) -> dict:
    """Public metadata: revision, license, gated flag. No token is sent."""
    url = HUB_API.format(dataset=urllib.parse.quote(dataset_id, safe="/"))
    payload = _http_json(url, timeout=timeout, retries=retries, sleep=sleep, opener=opener)
    card = payload.get("cardData") or {}
    tags = payload.get("tags") or []
    tag_license = next((tag.split(":", 1)[1] for tag in tags
                        if isinstance(tag, str) and tag.startswith("license:")), None)
    return {
        "id": payload.get("id", dataset_id),
        "sha": payload.get("sha"),
        "private": bool(payload.get("private", False)),
        "gated": payload.get("gated", False),
        "license": card.get("license") or tag_license,
        "last_modified": payload.get("lastModified"),
        "downloads": payload.get("downloads"),
    }


# -------------------------------------------------------------- FineWeb-Edu

class FineWebEdu:
    """FineWeb-Edu adapter. Revision is pinned for every stream."""

    name = "fineweb-edu"

    def __init__(self, config: str = "sample-10BT", revision: str | None = None,
                 backend: str = "auto", timeout: float = 60.0,
                 retries: int = 3, sleep=time.sleep,
                 opener=urllib.request.urlopen):
        spec = SOURCES[self.name]
        self.spec = spec
        self.config = config
        self.revision = revision
        self.backend = backend
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.opener = opener
        self._info = None

    def info(self) -> dict:
        if self._info is None:
            self._info = fetch_dataset_info(
                self.spec.dataset_id, timeout=self.timeout,
                retries=self.retries, sleep=self.sleep, opener=self.opener)
        return self._info

    def resolved_revision(self) -> str:
        if self.revision:
            return self.revision
        revision = self.info().get("sha")
        if not revision:
            raise SourceError("could not resolve a dataset revision")
        return revision

    def backend_name(self) -> str:
        if self.backend == "auto":
            try:
                import datasets  # noqa: F401
                return "datasets"
            except ImportError:
                return "http"
        if self.backend not in ("http", "datasets"):
            raise SourceError(f"unknown backend: {self.backend}")
        return self.backend

    def iter_documents(self, *, offset: int = 0, max_rows: int | None = None,
                       limit_rows: int | None = None):
        revision = self.resolved_revision()
        backend = self.backend_name()
        if backend == "datasets":
            yield from self._iter_datasets(revision, offset, max_rows, limit_rows)
        else:
            yield from self._iter_http(revision, offset, max_rows, limit_rows)

    # -- http ---------------------------------------------------------

    def _iter_http(self, revision: str, offset: int, max_rows: int | None,
                   limit_rows: int | None):
        spec = self.spec
        row = offset
        yielded = 0
        schema_checked = False
        while True:
            if max_rows is not None and row - offset >= max_rows:
                return
            length = PAGE_SIZE
            if max_rows is not None:
                length = min(length, max_rows - (row - offset))
            if limit_rows is not None:
                length = min(length, limit_rows - yielded)
            if length <= 0:
                return
            query = {
                "dataset": spec.dataset_id,
                "config": self.config,
                "split": spec.split,
                "offset": str(row),
                "length": str(length),
                "revision": revision,
            }
            url = ROWS_API + "?" + urllib.parse.urlencode(query)
            payload = _http_json(url, timeout=self.timeout, retries=self.retries,
                                 sleep=self.sleep, opener=self.opener)
            rows = payload.get("rows") or []
            features = [item.get("name") for item in payload.get("features") or []]
            if not schema_checked:
                for field_name in (spec.text_field, spec.id_field):
                    if field_name not in features:
                        raise SourceSchemaError(
                            f"{spec.dataset_id}/{self.config}: missing field "
                            f"{field_name!r}; got {features}")
                schema_checked = True
            if not rows:
                return
            for item in rows:
                data = item.get("row") or {}
                source_id = data.get(spec.id_field)
                text = data.get(spec.text_field)
                if source_id is None or not isinstance(text, str):
                    raise SourceSchemaError("row without id or text")
                meta = {
                    "language": data.get("language"),
                    "language_score": data.get("language_score"),
                    "int_score": data.get("int_score"),
                    "score": data.get("score"),
                    "token_count": data.get("token_count"),
                    "url": data.get("url"),
                    "dump": data.get("dump"),
                    "row_index": item.get("row_idx", row),
                }
                yield SourceDocument(
                    doc_key=f"{spec.dataset_id}@{revision}:{source_id}",
                    source=self.name,
                    dataset_id=spec.dataset_id,
                    revision=revision,
                    source_id=str(source_id),
                    text=text,
                    meta=meta,
                    arrival_index=int(item.get("row_idx", row)),
                )
                yielded += 1
                if limit_rows is not None and yielded >= limit_rows:
                    return
            row += len(rows)
            if len(rows) < length:
                return  # source exhausted

    # -- datasets -----------------------------------------------------

    def _iter_datasets(self, revision: str, offset: int, max_rows: int | None,
                       limit_rows: int | None):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise MissingDependencyError(
                "'datasets' is not installed; use --backend http or install it in "
                "a project environment (it is not part of the system Python)") from exc
        spec = self.spec
        dataset = load_dataset(spec.dataset_id, name=self.config, split=spec.split,
                               streaming=True, revision=revision)
        if offset:
            dataset = dataset.skip(offset)
        yielded = 0
        for index, row in enumerate(dataset):
            if max_rows is not None and index >= max_rows:
                return
            if limit_rows is not None and yielded >= limit_rows:
                return
            text = row.get(spec.text_field)
            source_id = row.get(spec.id_field)
            if source_id is None or not isinstance(text, str):
                raise SourceSchemaError("row without id or text")
            meta = {key: row.get(key) for key in
                    ("language", "language_score", "int_score", "score",
                     "token_count", "url", "dump")}
            meta["row_index"] = offset + index
            yield SourceDocument(
                doc_key=f"{spec.dataset_id}@{revision}:{source_id}",
                source=self.name,
                dataset_id=spec.dataset_id,
                revision=revision,
                source_id=str(source_id),
                text=text,
                meta=meta,
                arrival_index=offset + index,
            )
            yielded += 1


# ---------------------------------------------------------------- local JSONL

class LocalJsonlSource:
    """Offline adapter: one JSON object per line with ``id`` and ``text``.

    Used by tests and for user-provided corpora. ``offset`` is a line index.
    """

    name = "local-jsonl"

    def __init__(self, path: str, revision: str | None = None, **kwargs):
        if not os.path.exists(path):
            raise SourceError(f"local source not found: {path}")
        self.path = os.path.abspath(path)
        self.spec = SOURCES[self.name]
        self.revision = revision or f"local-{sha256_path(self.path)[:16]}"

    def iter_documents(self, *, offset: int = 0, max_rows: int | None = None,
                       limit_rows: int | None = None):
        yielded = 0
        with open(self.path, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < offset:
                    continue
                if max_rows is not None and index - offset >= max_rows:
                    return
                if limit_rows is not None and yielded >= limit_rows:
                    return
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                source_id = data.get("id", f"line-{index:08d}")
                text = data.get("text")
                if not isinstance(text, str):
                    raise SourceSchemaError(f"line {index} has no text")
                yield SourceDocument(
                    doc_key=f"{self.spec.dataset_id}@{self.revision}:{source_id}",
                    source=self.name,
                    dataset_id=self.spec.dataset_id,
                    revision=self.revision,
                    source_id=str(source_id),
                    text=text,
                    meta={"row_index": index, **(data.get("meta") or {})},
                    arrival_index=index,
                )
                yielded += 1


def sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_spec(name: str) -> SourceSpec:
    if name in PLANNED_SOURCES:
        raise SourceError(
            f"{name} is a documented extension point, not implemented yet; "
            f"see veylith/v3/DATA-SOURCES.md")
    if name not in SOURCES:
        raise SourceError(f"unknown source: {name}")
    return SOURCES[name]


def open_source(name: str, *, config: str | None = None, revision: str | None = None,
                backend: str = "auto", path: str | None = None,
                timeout: float = 60.0, retries: int = 3,
                sleep=time.sleep, opener=urllib.request.urlopen):
    spec = source_spec(name)
    if name == "local-jsonl":
        if not path:
            raise SourceError("local-jsonl requires --path")
        return LocalJsonlSource(path, revision=revision)
    if name == "fineweb-edu":
        return FineWebEdu(config=config or spec.config, revision=revision,
                          backend=backend, timeout=timeout, retries=retries,
                          sleep=sleep, opener=opener)
    raise SourceError(f"no adapter for {name}")
