"""Tests for code review fixes: exclude_empty, raise_on_error, MRO safety.

PowerScaleUnstructuredLoader/Reader tests below use FakePathLoader.
PowerScaleSimpleDirectoryReader tests use FakeFS or patch os.path.isdir locally
as needed, so no global os.path.isfile patch is required.
"""

from pathlib import Path
import pytest


# --- PowerScaleSimpleDirectoryReader exclude_empty -----------------------

def test_simple_directory_reader_exclude_empty_parameter_accepted(monkeypatch):
    """Test that exclude_empty parameter is accepted and stored."""
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    monkeypatch.setattr("os.path.isdir", lambda p: True)

    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        input_dir="/ifs/data",
        exclude_empty=True,
    )
    assert reader.exclude_empty is True


def test_simple_directory_reader_exclude_empty_defaults_to_false(monkeypatch):
    """Test that exclude_empty defaults to False, matching LlamaIndex."""
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    monkeypatch.setattr("os.path.isdir", lambda p: True)

    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        input_dir="/ifs/data",
    )
    assert reader.exclude_empty is False


def test_simple_directory_reader_exclude_empty_filters_zero_byte_files(monkeypatch):
    """Test that exclude_empty=True filters out zero-byte files."""
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)

    class FakeFS:
        def isdir(self, p):
            return p == "/ifs/data"

        def isfile(self, p):
            return p in ["/ifs/data/empty.txt", "/ifs/data/nonempty.txt"]

        def info(self, p):
            if p == "/ifs/data/empty.txt":
                return {"size": 0}
            return {"size": 100}

    fake_fs = FakeFS()
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        input_dir="/ifs/data",
        exclude_empty=True,
        fs=fake_fs,
    )

    assert reader._filter("/ifs/data/empty.txt") is False
    assert reader._filter("/ifs/data/nonempty.txt") is True


def test_simple_directory_reader_fs_parameter_accepted(monkeypatch):
    """Test that fs parameter is accepted and used for validation."""
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)

    class FakeFS:
        def isdir(self, p):
            return p == "/ifs/data"

        def isfile(self, p):
            return True

    fake_fs = FakeFS()
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        input_dir="/ifs/data",
        fs=fake_fs,
    )

    assert reader.fs is fake_fs


# --- PowerScaleUnstructuredLoader raise_on_error -------------------------

def test_unstructured_loader_raise_on_error_true_raises(monkeypatch):
    """Test that raise_on_error=True re-raises parse errors."""
    pytest.importorskip("langchain_unstructured")
    from langchain_unstructured import UnstructuredLoader
    from powerscale_rag_connector import PowerScaleUnstructuredLoader

    def exploding_lazy_load(self):
        raise RuntimeError("parse failed")
        yield  # pragma: no cover - keeps this a generator function

    monkeypatch.setattr(UnstructuredLoader, "lazy_load", exploding_lazy_load)

    loader = PowerScaleUnstructuredLoader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        folder_path="/ifs/data",
        raise_on_error=True,
    )

    # Replace path_loader with fake
    class FakePathLoader:
        def lazy_load(self):
            yield (Path("/ifs/data/a.txt"), 10, 1, [])

        def save_checkpoint(self):
            pass

    loader.path_loader = FakePathLoader()

    # Should raise
    with pytest.raises(RuntimeError, match="parse failed"):
        list(loader.lazy_load())



# --- PowerScaleUnstructuredReader raise_on_error -------------------------

def test_unstructured_reader_raise_on_error_true_raises(monkeypatch):
    """Test that raise_on_error=True re-raises parse errors."""
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.readers.file")
    from llama_index.readers.file import UnstructuredReader
    from powerscale_rag_connector import PowerScaleUnstructuredReader

    reader = PowerScaleUnstructuredReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        folder_path="/ifs/data",
        raise_on_error=True,
    )

    def exploding_load_data(self, **kwargs):
        raise RuntimeError("parse failed")

    monkeypatch.setattr(UnstructuredReader, "load_data", exploding_load_data)

    class FakePathLoader:
        def lazy_load(self):
            yield (Path("/ifs/data/a.txt"), 10, 1, [])

        def save_checkpoint(self):
            pass

    reader.path_loader = FakePathLoader()

    with pytest.raises(RuntimeError, match="parse failed"):
        list(reader.lazy_load_data())



# --- MRO safety test -----------------------------------------------------

def test_simple_directory_reader_init_does_not_perform_local_walk(monkeypatch):
    """Test that __init__ does not call SimpleDirectoryReader's filesystem walk."""
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    # If SimpleDirectoryReader.__init__ were called, it would call _add_files
    # and fail because /ifs/data is not a real local mount. We patch it to raise.
    def exploding_init(self, **kw):
        raise AssertionError("SimpleDirectoryReader.__init__ should not be called")

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", exploding_init)
    monkeypatch.setattr("os.path.isdir", lambda p: True)

    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        input_dir="/ifs/data",
    )

    assert reader.input_dir == Path("/ifs/data")
    assert reader.input_files == []


def test_simple_directory_reader_does_not_probe_fs_during_init(monkeypatch):
    """Initialization must not probe the filesystem for input_dir existence.

    Existence validation is the caller's responsibility: MetadataIQ owns file
    discovery and /ifs may not be mounted on the ingesting host.
    """
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)

    calls = []

    class RecordingFS:
        def isdir(self, p):
            calls.append(("isdir", p))
            return False

        def isfile(self, p):
            calls.append(("isfile", p))
            return False

    fake_fs = RecordingFS()

    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        input_dir="/ifs/data",
        fs=fake_fs,
    )

    assert reader.fs is fake_fs
    assert calls == [], f"filesystem was probed during init: {calls}"
