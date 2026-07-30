"""Tests for PowerScaleUnstructuredLoader and PowerScaleUnstructuredReader.

Both classes *subclass* their framework parser (``langchain_unstructured.UnstructuredLoader``
and ``llama_index.readers.file.UnstructuredReader``), so the tests patch the
inherited parse method rather than a module-level symbol. The PowerScale path
loader is faked so no filesystem or network access occurs.
"""

from pathlib import Path

import pytest


class FakePathLoader:
    """Replaces the internal PowerScalePathLoader; yields preset 4-tuples."""

    def __init__(self, tuples):
        self._tuples = tuples

    def lazy_load(self):
        yield from self._tuples

    def save_checkpoint(self):
        pass


# --- PowerScaleUnstructuredLoader (langchain) ---------------------------

def test_unstructured_loader_is_a_langchain_unstructured_loader():
    """The loader must derive from the framework class it wraps."""
    pytest.importorskip("langchain_unstructured")
    from langchain_core.document_loaders import BaseLoader
    from langchain_unstructured import UnstructuredLoader
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    assert issubclass(PowerScaleUnstructuredLoader, UnstructuredLoader)
    assert issubclass(PowerScaleUnstructuredLoader, BaseLoader)


def test_unstructured_loader_sets_metadata(monkeypatch):
    pytest.importorskip("langchain_unstructured")
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_unstructured import UnstructuredLoader
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    def fake_lazy_load(self):
        yield Document(page_content="chunk", metadata={})

    monkeypatch.setattr(UnstructuredLoader, "lazy_load", fake_lazy_load)

    loader = PowerScaleUnstructuredLoader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data"
    )
    loader.path_loader = FakePathLoader(
        [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    )

    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "/ifs/data/a.txt"
    assert docs[0].metadata["snapshot"] == 10
    assert docs[0].metadata["lin"] == 1
    assert docs[0].metadata["change_types"] == ["ENTRY_ADDED"]


def test_unstructured_loader_passes_chunking_strategy():
    """chunking_strategy must land in the inherited loader's partition kwargs."""
    pytest.importorskip("langchain_unstructured")
    pytest.importorskip("langchain_core")
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    loader = PowerScaleUnstructuredLoader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", chunking_strategy="basic",
    )
    assert loader.unstructured_kwargs.get("chunking_strategy") == "basic"


def test_unstructured_loader_forwards_extra_unstructured_kwargs():
    """Arbitrary UnstructuredLoader options must pass through to the parent."""
    pytest.importorskip("langchain_unstructured")
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    loader = PowerScaleUnstructuredLoader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", languages=["en", "de"], strategy="hi_res",
    )
    assert loader.unstructured_kwargs["languages"] == ["en", "de"]
    assert loader.unstructured_kwargs["strategy"] == "hi_res"


def test_unstructured_loader_retargets_file_path_per_file(monkeypatch):
    """One inherited loader instance is reused, retargeted at each file in turn.

    ``UnstructuredLoader.lazy_load()`` reads ``self.file_path`` at call time and
    holds no other per-file state, so reassigning it is sufficient and avoids
    rebuilding the Unstructured client for every file.
    """
    pytest.importorskip("langchain_unstructured")
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_unstructured import UnstructuredLoader
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    seen = []

    def fake_lazy_load(self):
        seen.append((self.file_path, dict(self.unstructured_kwargs)))
        yield Document(page_content="chunk", metadata={})

    monkeypatch.setattr(UnstructuredLoader, "lazy_load", fake_lazy_load)

    loader = PowerScaleUnstructuredLoader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        folder_path="/ifs/data",
        chunking_strategy="basic",
    )
    loader.path_loader = FakePathLoader(
        [
            (Path("/ifs/data/a.txt"), 10, 1, []),
            (Path("/ifs/data/b.txt"), 11, 2, []),
        ]
    )

    list(loader.lazy_load())
    assert seen == [
        ("/ifs/data/a.txt", {"chunking_strategy": "basic"}),
        ("/ifs/data/b.txt", {"chunking_strategy": "basic"}),
    ]


def test_unstructured_loader_skips_parser_error_when_false(monkeypatch):
    pytest.importorskip("langchain_unstructured")
    from langchain_unstructured import UnstructuredLoader
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    def exploding_lazy_load(self):
        raise RuntimeError("parse failed")
        yield  # pragma: no cover - keeps this a generator function

    monkeypatch.setattr(UnstructuredLoader, "lazy_load", exploding_lazy_load)
    loader = PowerScaleUnstructuredLoader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data",
        raise_on_error=False,
    )
    loader.path_loader = FakePathLoader([(Path("/ifs/data/a.txt"), 10, 1, [])])
    # error is logged and skipped => empty result, no raise
    assert list(loader.lazy_load()) == []


# --- PowerScaleUnstructuredReader (llama_index) -------------------------

def test_unstructured_reader_is_a_llamaindex_unstructured_reader():
    """The reader must derive from the framework class it wraps."""
    pytest.importorskip("llama_index.readers.file")
    from llama_index.core.readers.base import BaseReader
    from llama_index.readers.file import UnstructuredReader
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    assert issubclass(PowerScaleUnstructuredReader, UnstructuredReader)
    assert issubclass(PowerScaleUnstructuredReader, BaseReader)


def test_unstructured_reader_mode_maps_to_split_documents():
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    single = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", mode="single",
    )
    elements = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", mode="elements",
    )
    assert single._PowerScaleUnstructuredReader__split_documents is False
    assert elements._PowerScaleUnstructuredReader__split_documents is True


def test_unstructured_reader_default_languages():
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    reader = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data"
    )
    assert reader._PowerScaleUnstructuredReader__languages == ["en"]


def test_unstructured_reader_sets_metadata(monkeypatch):
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from llama_index.core import Document
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    from llama_index.readers.file import UnstructuredReader

    reader = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", mode="elements",
    )

    def fake_load_data(self, file=None, unstructured_kwargs=None, **kwargs):
        languages = (unstructured_kwargs or {}).get("languages")
        return [Document(text="parsed", metadata={"languages": languages})]

    monkeypatch.setattr(UnstructuredReader, "load_data", fake_load_data)
    reader.path_loader = FakePathLoader(
        [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_MODIFIED"])]
    )

    docs = list(reader.lazy_load_data())
    assert docs[0].metadata["languages"] == ["en"]
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "/ifs/data/a.txt"
    assert docs[0].metadata["snapshot"] == 10
    assert docs[0].metadata["lin"] == 1
    assert docs[0].metadata["change_types"] == ["ENTRY_MODIFIED"]


def test_unstructured_reader_skips_parser_error_when_false(monkeypatch):
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    from llama_index.readers.file import UnstructuredReader

    reader = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data",
        raise_on_error=False,
    )

    def exploding_load_data(self, **kwargs):
        raise RuntimeError("parse failed")

    monkeypatch.setattr(UnstructuredReader, "load_data", exploding_load_data)
    reader.path_loader = FakePathLoader([(Path("/ifs/data/a.txt"), 10, 1, [])])
    assert list(reader.lazy_load_data()) == []


def test_unstructured_reader_parses_each_file_via_inherited_load_data(monkeypatch):
    """Each discovered file is parsed through the inherited single-file load_data."""
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from llama_index.core import Document
    from llama_index.readers.file import UnstructuredReader
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    seen = []

    def fake_load_data(self, file=None, split_documents=None, **kwargs):
        seen.append((str(file), split_documents))
        return [Document(text="chunk", metadata={})]

    monkeypatch.setattr(UnstructuredReader, "load_data", fake_load_data)

    reader = PowerScaleUnstructuredReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        folder_path="/ifs/data",
        mode="elements",
    )
    reader.path_loader = FakePathLoader([
        (Path("/ifs/data/a.txt"), 10, 1, []),
        (Path("/ifs/data/b.txt"), 11, 2, []),
    ])

    list(reader.lazy_load_data())
    assert seen == [("/ifs/data/a.txt", True), ("/ifs/data/b.txt", True)]


def test_unstructured_reader_load_data_with_file_delegates_to_parent(monkeypatch):
    """Passing ``file=`` must keep the upstream UnstructuredReader contract.

    The PowerScale scan must not run in that case.
    """
    pytest.importorskip("llama_index.readers.file")
    from llama_index.core import Document
    from llama_index.readers.file import UnstructuredReader
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    def fake_load_data(self, file=None, **kwargs):
        return [Document(text=f"parent:{file}", metadata={})]

    monkeypatch.setattr(UnstructuredReader, "load_data", fake_load_data)

    reader = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data",
    )

    def fail():
        raise AssertionError("PowerScale scan must not run when file= is given")

    reader.path_loader = type(
        "Boom", (), {"lazy_load": lambda self: fail(), "save_checkpoint": lambda self: None}
    )()

    docs = reader.load_data(file=Path("/ifs/data/explicit.txt"))
    assert [d.text for d in docs] == ["parent:/ifs/data/explicit.txt"]
