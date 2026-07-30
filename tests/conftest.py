"""Shared pytest fixtures and fakes for the PowerScale RAG Connector test suite.

The whole suite runs without a live Elasticsearch cluster. `FakeElasticsearch`
implements just enough of the elasticsearch-py 8.x client surface used by
`PowerScaleHelper` (``get``, ``search``, ``index``) so behaviour can be asserted
deterministically.
"""

import copy
import importlib.util
import os
import subprocess
import sys

import pytest

# Support running against the src/ layout without an editable install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from elasticsearch import exceptions  # noqa: E402


# Optional test dependencies. (package name, import to check)
_OPTIONAL_DEPS = [
    ("langchain-core", "langchain_core"),
    ("langchain-unstructured", "langchain_unstructured"),
    ("llama-index", "llama_index.core"),
    ("llama-index-readers-file", "llama_index.readers.file"),
    ("unstructured", "unstructured"),
]


def pytest_addoption(parser):
    parser.addoption(
        "--install-extras",
        action="store_true",
        default=False,
        help="Auto-install optional framework dependencies (langchain, llama-index, unstructured) at session start.",
    )


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def pytest_sessionstart(session):
    """Install optional test dependencies if --install-extras was passed."""
    if not session.config.getoption("--install-extras", default=False):
        return

    missing = [
        pkg for pkg, mod in _OPTIONAL_DEPS if not _module_available(mod)
    ]
    if not missing:
        return

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(
            f"Warning: could not auto-install optional packages {missing}: {e}",
            file=sys.stderr,
        )


def make_not_found() -> Exception:
    """Build an elasticsearch NotFoundError across client minor-version signatures."""
    for kwargs in ({"meta": None, "body": None}, {}):
        try:
            return exceptions.NotFoundError("not found", **kwargs)  # type: ignore[arg-type]
        except TypeError:
            continue
    # Last resort: a bare instance via __new__ so isinstance checks still work.
    return exceptions.NotFoundError.__new__(exceptions.NotFoundError)


def make_conflict() -> Exception:
    """Build an elasticsearch ConflictError across client minor-version signatures."""
    for kwargs in ({"meta": None, "body": None}, {}):
        try:
            return exceptions.ConflictError("conflict", **kwargs)  # type: ignore[arg-type]
        except TypeError:
            continue
    return exceptions.ConflictError.__new__(exceptions.ConflictError)


def make_hit(path, lin, snapshot, change_types=None, btime=0, mtime=0):
    """Construct a single MetadataIQ-style search hit."""
    return {
        "_source": {
            "data": {
                "path": path,
                "lin": lin,
                "change_types": list(change_types) if change_types is not None else [],
                "btime": btime,
                "mtime": mtime,
            },
            "metadata": {"snapshots": {"s2": snapshot}},
        },
        "sort": [lin],
    }


class FakeElasticsearch:
    """Minimal fake implementing the client methods PowerScaleHelper relies on."""

    DATASET_INDEX = "powerscale_rag_datasets"

    def __init__(
        self,
        *,
        checkpoint_doc=None,
        max_snapid=None,
        search_pages=None,
        dataset_doc=None,
        raise_on_agg=None,
        raise_on_search=None,
        seq_no=None,
        primary_term=None,
        conflicts=0,
    ):
        # checkpoint_doc None => NotFoundError on checkpoint get (first run)
        self.checkpoint_doc = checkpoint_doc
        self.max_snapid = max_snapid
        # search_pages: list of pages, each page a list of hits. Consumed in order.
        self.search_pages = [list(p) for p in (search_pages or [])]
        self.dataset_doc = dataset_doc
        self.raise_on_agg = raise_on_agg
        self.raise_on_search = raise_on_search
        # Document version returned by get(); enables optimistic concurrency control.
        self.seq_no = seq_no
        self.primary_term = primary_term
        # Number of index() calls that should raise ConflictError before succeeding.
        self.conflicts = conflicts

        self.indexed = []          # records of index() writes
        self.get_calls = []        # (index, id) tuples
        self.search_calls = []     # kwargs of each search() call
        self._page_idx = 0

    # -- elasticsearch client surface --------------------------------------
    def get(self, index, id):
        self.get_calls.append((index, id))
        if index == self.DATASET_INDEX:
            if self.dataset_doc is None:
                raise make_not_found()
            return {"_source": self.dataset_doc}
        if self.checkpoint_doc is None:
            raise make_not_found()
        resp = {"_source": copy.deepcopy(self.checkpoint_doc)}
        if self.seq_no is not None:
            resp["_seq_no"] = self.seq_no
        if self.primary_term is not None:
            resp["_primary_term"] = self.primary_term
        return resp

    def search(self, index, **kwargs):
        self.search_calls.append(kwargs)
        if "aggs" in kwargs:
            if self.raise_on_agg is not None:
                raise self.raise_on_agg
            return {"aggregations": {"max_snapid": {"value": self.max_snapid}}}
        # paged search_after query
        if self.raise_on_search is not None:
            raise self.raise_on_search
        if self._page_idx < len(self.search_pages):
            hits = self.search_pages[self._page_idx]
            self._page_idx += 1
        else:
            hits = []
        return {"hits": {"hits": hits}}

    def index(self, index, id, document, **kwargs):
        if self.conflicts > 0:
            self.conflicts -= 1
            # Simulate another writer having advanced the document.
            if self.seq_no is not None:
                self.seq_no += 1
            raise make_conflict()
        self.indexed.append(
            {
                "index": index,
                "id": id,
                "document": copy.deepcopy(document),
                "kwargs": dict(kwargs),
            }
        )
        if self.seq_no is not None:
            self.seq_no += 1
            return {"_seq_no": self.seq_no, "_primary_term": self.primary_term}
        return None


@pytest.fixture
def make_helper(monkeypatch):
    """Factory that patches the Elasticsearch constructor and builds a helper.

    Usage:
        helper = make_helper(fake_es, folder_path="/ifs/data")
    """
    import importlib

    from powerscale_rag_connector import PowerScaleHelper

    # NOTE: `powerscale_rag_connector.PowerScaleHelper` resolves to the *class*
    # (re-exported in __init__), so fetch the submodule from sys.modules to patch
    # its module-level Elasticsearch symbol.
    helper_module = importlib.import_module(
        "powerscale_rag_connector.PowerScaleHelper"
    )

    def _make(fake_es, **kwargs):
        monkeypatch.setattr(
            helper_module, "Elasticsearch", lambda *a, **k: fake_es
        )
        params = {
            "es_host_url": "http://localhost:9200",
            "es_index_name": "idx",
            "es_api_key": "secret-key",
        }
        params.update(kwargs)
        return PowerScaleHelper(**params)

    return _make


class FakeHelper:
    """Stand-in for PowerScaleHelper used by loader/reader tests.

    Records the snapshot_id passed to get_directory_changes and replays a
    preconfigured list of 4-tuples.
    """

    def __init__(self, tuples=None):
        self.tuples = list(tuples or [])
        self.calls = []
        self.save_calls = []

    def get_directory_changes(self, snapshot_id=-1, save_checkpoint=True):
        self.calls.append(snapshot_id)
        for t in self.tuples:
            yield t

    def save_checkpoint(self):
        self.save_calls.append(True)
