"""Tests for PowerScalePathLoader and PowerScaleDocumentLoader.

Both loaders defer to a lazily-created PowerScaleHelper. Tests inject a
FakeHelper via the name-mangled ``__pshelper`` slot to avoid touching
Elasticsearch.
"""

from pathlib import Path

import pytest

from tests.conftest import FakeHelper

from powerscale_rag_connector import PowerScalePathLoader


TUPLES = [
    (Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"]),
    (Path("/ifs/data/b.txt"), 11, 2, ["ENTRY_MODIFIED"]),
]


def _make_path_loader(fake_helper, **kwargs):
    loader = PowerScalePathLoader(
        es_host_url="http://localhost:9200",
        es_index_name="idx",
        es_api_key="key",
        folder_path="/ifs/data",
        **kwargs,
    )
    loader._PowerScalePathLoader__pshelper = fake_helper
    return loader


# --- PowerScalePathLoader ------------------------------------------------

def test_path_loader_yields_tuples_unchanged():
    fake = FakeHelper(TUPLES)
    loader = _make_path_loader(fake)
    assert list(loader.lazy_load()) == TUPLES


def test_path_loader_normal_uses_default_snapshot():
    fake = FakeHelper(TUPLES)
    loader = _make_path_loader(fake, force_scan=False)
    list(loader.lazy_load())
    assert fake.calls == [-1]


def test_path_loader_force_scan_uses_zero():
    fake = FakeHelper(TUPLES)
    loader = _make_path_loader(fake, force_scan=True)
    list(loader.lazy_load())
    assert fake.calls == [0]


def test_path_loader_yields_files_without_local_existence_check(monkeypatch):
    """PowerScalePathLoader should not skip MetadataIQ entries based on local file existence."""
    fake = FakeHelper(TUPLES)
    loader = _make_path_loader(fake)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    results = list(loader.lazy_load())
    assert results == TUPLES


# --- PowerScaleDocumentLoader (requires langchain-core) ------------------

def test_document_loader_yields_documents():
    pytest.importorskip("langchain_core")
    from powerscale_rag_connector import PowerScaleDocumentLoader

    fake = FakeHelper(TUPLES)
    loader = PowerScaleDocumentLoader(
        es_host_url="http://localhost:9200",
        es_index_name="idx",
        es_api_key="key",
        folder_path="/ifs/data",
    )
    loader._PowerScaleDocumentLoader__pshelper = fake

    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].page_content == ""
    assert docs[0].metadata == {
        "source": "/ifs/data/a.txt",
        "snapshot": 10,
        "lin": 1,
        "change_types": ["ENTRY_ADDED"],
    }
    assert docs[1].metadata["change_types"] == ["ENTRY_MODIFIED"]


def test_document_loader_force_scan_uses_zero():
    pytest.importorskip("langchain_core")
    from powerscale_rag_connector import PowerScaleDocumentLoader

    fake = FakeHelper(TUPLES)
    loader = PowerScaleDocumentLoader(
        es_host_url="http://localhost:9200",
        es_index_name="idx",
        es_api_key="key",
        folder_path="/ifs/data",
        force_scan=True,
    )
    loader._PowerScaleDocumentLoader__pshelper = fake
    list(loader.lazy_load())
    assert fake.calls == [0]


class _SavingFakeHelper:
    """Fake that records save_checkpoint calls from get_directory_changes."""

    def __init__(self, tuples):
        self.tuples = list(tuples)
        self.calls = []
        self.save_calls = []

    def get_directory_changes(self, snapshot_id=-1, save_checkpoint=True):
        self.calls.append((snapshot_id, save_checkpoint))
        for t in self.tuples:
            yield t
        if save_checkpoint:
            self.save_calls.append(True)

    def save_checkpoint(self):
        self.save_calls.append("explicit")


def test_document_loader_auto_commits_checkpoint_on_full_consumption():
    """Checkpoint is saved automatically when lazy_load() is fully consumed."""
    pytest.importorskip("langchain_core")
    from powerscale_rag_connector import PowerScaleDocumentLoader

    fake = _SavingFakeHelper(TUPLES)
    loader = PowerScaleDocumentLoader(
        es_host_url="http://localhost:9200",
        es_index_name="idx",
        es_api_key="key",
        folder_path="/ifs/data",
    )
    loader._PowerScaleDocumentLoader__pshelper = fake

    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert fake.save_calls == [True]


def test_document_loader_does_not_save_checkpoint_on_early_break():
    """If the caller breaks out of the generator, the checkpoint must not advance."""
    pytest.importorskip("langchain_core")
    from powerscale_rag_connector import PowerScaleDocumentLoader

    fake = _SavingFakeHelper(TUPLES)
    loader = PowerScaleDocumentLoader(
        es_host_url="http://localhost:9200",
        es_index_name="idx",
        es_api_key="key",
        folder_path="/ifs/data",
    )
    loader._PowerScaleDocumentLoader__pshelper = fake

    for doc in loader.lazy_load():
        break  # consume only the first document

    assert fake.save_calls == []
