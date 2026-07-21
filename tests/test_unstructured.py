"""Tests for PowerScaleUnstructuredLoader and PowerScaleUnstructuredReader.

These wrap third-party parsers; tests inject fakes for both the PowerScale path
loader and the underlying parser so no filesystem or network access occurs.
The internal PowerScalePathLoader now owns the missing-file existence check,
so the unstructured loader/reader tests do not need to patch os.path.isfile.
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

def test_unstructured_loader_sets_metadata(monkeypatch):
    pytest.importorskip("langchain_unstructured")
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from powerscale_rag_connector import PowerScaleUnstructuredLoader
    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredLoader")

    class FakeUnstructuredLoader:
        def __init__(self, file_path=None, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def lazy_load(self):
            yield Document(page_content="chunk", metadata={})

    monkeypatch.setattr(mod, "UnstructuredLoader", FakeUnstructuredLoader)

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


def test_unstructured_loader_passes_chunking_strategy(monkeypatch):
    pytest.importorskip("langchain_unstructured")
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from powerscale_rag_connector import PowerScaleUnstructuredLoader
    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredLoader")

    captured = {}

    class FakeUnstructuredLoader:
        def __init__(self, file_path=None, **kwargs):
            captured.update(kwargs)

        def lazy_load(self):
            yield Document(page_content="c", metadata={})

    monkeypatch.setattr(mod, "UnstructuredLoader", FakeUnstructuredLoader)

    loader = PowerScaleUnstructuredLoader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", chunking_strategy="basic",
    )
    loader.path_loader = FakePathLoader([(Path("/ifs/data/a.txt"), 10, 1, [])])
    list(loader.lazy_load())
    assert captured.get("chunking_strategy") == "basic"


def test_unstructured_loader_creates_loader_per_file(monkeypatch):
    pytest.importorskip("langchain_unstructured")
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from powerscale_rag_connector import PowerScaleUnstructuredLoader
    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredLoader")

    calls = []

    class FakeUnstructuredLoader:
        def __init__(self, file_path=None, **kwargs):
            calls.append((file_path, kwargs))

        def lazy_load(self):
            yield Document(page_content="chunk", metadata={})

    monkeypatch.setattr(mod, "UnstructuredLoader", FakeUnstructuredLoader)

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
    assert len(calls) == 2
    assert calls[0] == ("/ifs/data/a.txt", {"chunking_strategy": "basic"})
    assert calls[1] == ("/ifs/data/b.txt", {"chunking_strategy": "basic"})


def test_unstructured_loader_skips_parser_error_when_false(monkeypatch):
    pytest.importorskip("langchain_unstructured")
    from powerscale_rag_connector import PowerScaleUnstructuredLoader
    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredLoader")

    class ExplodingLoader:
        def __init__(self, file_path=None, **kwargs):
            pass

        def lazy_load(self):
            raise RuntimeError("parse failed")

    monkeypatch.setattr(mod, "UnstructuredLoader", ExplodingLoader)
    loader = PowerScaleUnstructuredLoader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data",
        raise_on_error=False,
    )
    loader.path_loader = FakePathLoader([(Path("/ifs/data/a.txt"), 10, 1, [])])
    # error is logged and skipped => empty result, no raise
    assert list(loader.lazy_load()) == []


# --- PowerScaleUnstructuredReader (llama_index) -------------------------

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

    reader = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        folder_path="/ifs/data", mode="elements",
    )

    class FakeReader:
        def load_data(self, file=None, split_documents=None, unstructured_kwargs=None):
            languages = (unstructured_kwargs or {}).get("languages")
            return [Document(text="parsed", metadata={"languages": languages})]

    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredReader")
    monkeypatch.setattr(mod, "UnstructuredReader", FakeReader)
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

    reader = PowerScaleUnstructuredReader(
        es_host_url="h", es_index_name="i", es_api_key="k", folder_path="/ifs/data",
        raise_on_error=False,
    )

    class ExplodingReader:
        def load_data(self, **kwargs):
            raise RuntimeError("parse failed")

    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredReader")
    monkeypatch.setattr(mod, "UnstructuredReader", ExplodingReader)
    reader.path_loader = FakePathLoader([(Path("/ifs/data/a.txt"), 10, 1, [])])
    assert list(reader.lazy_load_data()) == []


def test_unstructured_reader_creates_reader_per_file(monkeypatch):
    """PowerScaleUnstructuredReader must create a new UnstructuredReader per file
    to avoid retained state leaking between files.
    """
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from powerscale_rag_connector import PowerScaleUnstructuredReader
    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredReader")

    calls = []

    class FakeUnstructuredReader:
        def __init__(self):
            calls.append("init")

        def load_data(self, file=None, split_documents=None, unstructured_kwargs=None):
            from llama_index.core import Document
            return [Document(text="chunk", metadata={})]

    monkeypatch.setattr(mod, "UnstructuredReader", FakeUnstructuredReader)

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
    assert len(calls) == 2
