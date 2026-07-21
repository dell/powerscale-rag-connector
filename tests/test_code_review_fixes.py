"""Tests for code review fixes: exclude_empty, raise_on_error, MRO safety.

PowerScaleUnstructuredLoader/Reader no longer perform their own os.path.isfile
checks; their tests below use FakePathLoader. PowerScaleSimpleDirectoryReader
tests use FakeFS or patch os.path.isdir/os.path.isfile locally as needed, so no
global os.path.isfile patch is required.
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
    from powerscale_rag_connector import PowerScaleUnstructuredReader
    import importlib
    mod = importlib.import_module("powerscale_rag_connector.PowerScaleUnstructuredReader")

    reader = PowerScaleUnstructuredReader(
        es_host_url="h",
        es_index_name="i",
        es_api_key="k",
        folder_path="/ifs/data",
        raise_on_error=True,
    )

    class ExplodingReader:
        def load_data(self, **kwargs):
            raise RuntimeError("parse failed")

    monkeypatch.setattr(mod, "UnstructuredReader", ExplodingReader)

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


def test_simple_directory_reader_fs_used_for_validation(monkeypatch):
    """Test that fs is used for existence checks during initialization."""
    pytest.importorskip("llama_index.core")
    from llama_index.core import SimpleDirectoryReader
    from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)

    class FakeFS:
        def isdir(self, p):
            return False  # Simulate directory not existing

        def isfile(self, p):
            return True

    fake_fs = FakeFS()

    with pytest.raises(ValueError, match="Directory does not exist"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h",
            es_index_name="i",
            es_api_key="k",
            input_dir="/ifs/data",
            fs=fake_fs,
        )
