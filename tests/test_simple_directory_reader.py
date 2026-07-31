"""Tests for PowerScaleSimpleDirectoryReader.

The base LlamaIndex SimpleDirectoryReader performs real filesystem work in its
constructor, so most tests build the object via ``__new__`` and set only the
attributes the method under test needs. Constructor-validation tests patch the
base ``__init__`` to a no-op.
"""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.file.base import get_default_fs

from powerscale_rag_connector import PowerScaleSimpleDirectoryReader
from tests.conftest import FakeHelper


def _bare_reader(**attrs):
    """Instantiate without running SimpleDirectoryReader.__init__."""
    reader = PowerScaleSimpleDirectoryReader.__new__(PowerScaleSimpleDirectoryReader)
    defaults = {
        "_exclude_exact": set(),
        "required_exts": None,
        "exclude_hidden": False,
        "exclude_empty": False,
        "_orig_cb": None,
        "_explicit_files": None,
        "_input_dir": None,
        "recursive": True,
        "fs": get_default_fs(),
    }
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(reader, k, v)
    return reader


# --- _is_hidden ----------------------------------------------------------

def test_is_hidden_true_for_dotfile():
    reader = _bare_reader()
    assert reader._is_hidden("/ifs/data/.secret/file.txt") is True


def test_is_hidden_false_for_normal_path():
    reader = _bare_reader()
    assert reader._is_hidden("/ifs/data/dir/file.txt") is False


# --- _filter -------------------------------------------------------------

def test_filter_excludes_exact_match(monkeypatch):
    reader = _bare_reader(_exclude_exact={"/ifs/data/skip.txt"})
    assert reader._filter("/ifs/data/skip.txt") is False


def test_filter_excludes_glob_pattern(monkeypatch):
    reader = _bare_reader(_exclude_exact={"*.txt"})
    assert reader._filter("/ifs/data/a.txt") is False
    assert reader._filter("/ifs/data/subdir/b.txt") is False
    assert reader._filter("/ifs/data/a.pdf") is True


def test_filter_excludes_recursive_double_star_pattern(monkeypatch):
    reader = _bare_reader(_exclude_exact={"/ifs/data/**/temp.*"})
    assert reader._filter("/ifs/data/subdir/temp.txt") is False
    assert reader._filter("/ifs/data/temp.txt") is False
    assert reader._filter("/ifs/data/subdir/temp.log") is False
    assert reader._filter("/ifs/data/subdir/not.txt") is True


def test_filter_excludes_relative_recursive_double_star_pattern(monkeypatch):
    reader = _bare_reader(_exclude_exact={"**/temp.*"})
    assert reader._filter("/ifs/data/subdir/temp.txt") is False
    assert reader._filter("/ifs/data/temp.log") is False
    assert reader._filter("/ifs/data/subdir/not.txt") is True


def test_filter_accepts_missing_file(monkeypatch):
    """File existence is not verified at the reader/loader level."""
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    reader = _bare_reader()
    assert reader._filter("/ifs/data/gone.txt") is True


def test_filter_required_exts(monkeypatch):
    reader = _bare_reader(required_exts={".pdf"})
    assert reader._filter("/ifs/data/doc.pdf") is True
    assert reader._filter("/ifs/data/doc.txt") is False


def test_filter_required_exts_normalizes_without_dot(monkeypatch):
    reader = _bare_reader(required_exts={"pdf"})
    assert reader._filter("/ifs/data/doc.pdf") is True


def test_filter_excludes_hidden(monkeypatch):
    reader = _bare_reader(exclude_hidden=True)
    assert reader._filter("/ifs/data/.hidden/file.txt") is False


def test_filter_accepts_normal_file(monkeypatch):
    reader = _bare_reader()
    assert reader._filter("/ifs/data/file.txt") is True


# --- _merge_metadata -----------------------------------------------------

def test_merge_metadata_basic():
    reader = _bare_reader()
    selected = {"/ifs/data/a.txt": (10, 1, ["ENTRY_ADDED"])}
    meta = reader._merge_metadata("/ifs/data/a.txt", selected=selected)
    assert meta == {
        "source": "/ifs/data/a.txt",
        "snapshot": 10,
        "lin": 1,
        "change_types": ["ENTRY_ADDED"],
    }


def test_merge_metadata_includes_user_callback():
    reader = _bare_reader(_orig_cb=lambda p: {"custom": "value"})
    selected = {"/ifs/data/a.txt": (10, 1, ["ENTRY_ADDED"])}
    meta = reader._merge_metadata("/ifs/data/a.txt", selected=selected)
    assert meta["custom"] == "value"
    assert meta["lin"] == 1


def test_merge_metadata_user_callback_cannot_overwrite_ps_fields():
    """PowerScale structural fields (lin, snapshot, etc.) always win over callback output."""
    reader = _bare_reader(_orig_cb=lambda p: {"lin": 999, "snapshot": 999, "source": "bad", "custom": "ok"})
    selected = {"/ifs/data/a.txt": (10, 1, ["ENTRY_ADDED"])}
    meta = reader._merge_metadata("/ifs/data/a.txt", selected=selected)
    # PowerScale values must not be overwritten by the callback
    assert meta["lin"] == 1
    assert meta["snapshot"] == 10
    assert meta["source"] == "/ifs/data/a.txt"
    # Custom keys from the callback are still present
    assert meta["custom"] == "ok"


def test_merge_metadata_user_callback_error_logs_warning(caplog):
    def boom(_p):
        raise RuntimeError("bad callback")

    reader = _bare_reader(_orig_cb=boom)
    selected = {"/ifs/data/a.txt": (10, 1, [])}
    with caplog.at_level("WARNING"):
        meta = reader._merge_metadata("/ifs/data/a.txt", selected=selected)
    assert meta["source"] == "/ifs/data/a.txt"
    assert "bad callback" in caplog.text


# --- lazy_load_data ------------------------------------------------------

class _FakeChildReader:
    def __init__(self, docs):
        self._docs = docs

    def iter_data(self, show_progress=False):
        # Mimics SimpleDirectoryReader.iter_data(): yields one list per file.
        # For tests each fake child is a single list of docs.
        yield list(self._docs)

    def load_data(self, show_progress=False, num_workers=None, fs=None):
        return list(self._docs)

    def lazy_load_data(self, **kwargs):
        # Kept for backward compatibility with any older tests.
        yield from self._docs


def _reader_for_lazy(monkeypatch, tuples, child_docs, **attrs):
    reader = _bare_reader(
        _force_scan=False,
        **attrs,
    )
    reader._PowerScaleSimpleDirectoryReader__pshelper = FakeHelper(tuples)
    monkeypatch.setattr(
        reader, "_child_reader", lambda files, meta_wrap: _FakeChildReader(child_docs)
    )
    return reader


def test_lazy_load_data_empty_returns_nothing(monkeypatch):
    reader = _reader_for_lazy(monkeypatch, tuples=[], child_docs=[])
    assert list(reader.lazy_load_data()) == []


def test_lazy_load_data_yields_child_docs(monkeypatch):
    tuples = [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    docs = ["doc-a"]
    reader = _reader_for_lazy(monkeypatch, tuples=tuples, child_docs=docs)
    assert list(reader.lazy_load_data()) == ["doc-a"]


def test_lazy_load_data_force_scan_passes_zero(monkeypatch):
    fake = FakeHelper([(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])])
    reader = _bare_reader(_force_scan=True)
    reader._PowerScaleSimpleDirectoryReader__pshelper = fake
    monkeypatch.setattr(
        reader, "_child_reader", lambda files, meta_wrap: _FakeChildReader(["x"])
    )
    list(reader.lazy_load_data())
    assert fake.calls == [0]


# --- constructor validation ---------------------------------------------

def test_init_requires_exactly_one_scope(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="exactly one"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k"
        )


def test_init_input_dir_must_start_with_ifs(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="must start with '/ifs'"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k",
            input_dir="/mnt/data",
        )


def test_init_input_dir_not_checked_against_local_filesystem(monkeypatch):
    """input_dir must not be validated against the local filesystem.

    File discovery is delegated to MetadataIQ, so the connector has to work from
    hosts where /ifs is not mounted.
    """
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    monkeypatch.setattr("os.path.isdir", lambda p: False)
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        input_dir="/ifs/missing",
    )
    assert reader._input_dir == "/ifs/missing"


def test_init_input_files_validation(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="must start with '/ifs'"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k",
            input_files=["/tmp/x.txt"],
        )


def test_init_rejects_both_input_dir_and_input_files(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="exactly one"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k",
            input_dir="/ifs/data",
            input_files=["/ifs/data/a.txt"],
        )


def test_init_input_files_none_raises(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="exactly one"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k",
            input_dir=None,
            input_files=None,
        )


def test_init_input_files_empty_list_raises(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="cannot be empty"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k",
            input_files=[],
        )


def test_init_input_files_valid_succeeds(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        input_files=["/ifs/data/a.txt"],
    )
    assert reader._explicit_files == ["/ifs/data/a.txt"]


# --- base attributes -----------------------------------------------------

def test_init_sets_base_attributes_for_input_dir(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        input_dir="/ifs/data",
    )
    assert reader.input_dir == Path("/ifs/data")
    assert reader.input_files == []
    assert reader.fs is not None
    assert reader.file_metadata is not None
    assert reader.exclude is None
    assert reader.recursive is True


def test_init_sets_base_attributes_for_input_files(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        input_files=["/ifs/data/a.txt"],
    )
    assert reader.input_dir is None
    assert reader.input_files == [Path("/ifs/data/a.txt")]
    assert reader.fs is not None
    assert reader.file_metadata is not None


# --- load_data / iter_data / list_resources / aload_data -----------------

def _reader_for_load_tests(monkeypatch, tuples, child_docs, **attrs):
    """Build a reader with a fake helper and a fake child for load/iter tests."""
    reader = _bare_reader(
        _force_scan=False,
        **attrs,
    )
    fake = FakeHelper(tuples)
    reader._PowerScaleSimpleDirectoryReader__pshelper = fake
    monkeypatch.setattr(
        reader, "_child_reader", lambda files, meta_wrap: _FakeChildReader(child_docs)
    )
    return reader, fake


def test_load_data_returns_docs_and_saves_checkpoint(monkeypatch):
    tuples = [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=tuples, child_docs=["doc-a"])
    assert reader.load_data() == ["doc-a"]
    assert len(fake.save_calls) == 1


def test_iter_data_returns_lists_and_saves_checkpoint(monkeypatch):
    tuples = [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=tuples, child_docs=["doc-a"])
    assert list(reader.iter_data()) == [["doc-a"]]
    assert len(fake.save_calls) == 1


def test_iter_data_early_break_does_not_save_checkpoint(monkeypatch):
    tuples = [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=tuples, child_docs=["doc-a"])
    gen = reader.iter_data()
    next(gen)
    gen.close()
    assert len(fake.save_calls) == 0


def test_list_resources_returns_all_files(monkeypatch):
    tuples = [
        (Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"]),
        (Path("/ifs/data/b.txt"), 10, 2, ["ENTRY_ADDED"]),
        (Path("/ifs/data/c.txt"), 10, 3, ["ENTRY_ADDED"]),
    ]
    reader, fake = _reader_for_load_tests(
        monkeypatch, tuples=tuples, child_docs=["doc"]
    )
    assert reader.list_resources() == [
        "/ifs/data/a.txt",
        "/ifs/data/b.txt",
        "/ifs/data/c.txt",
    ]
    assert len(fake.save_calls) == 0


def test_aload_data_returns_docs_and_saves_checkpoint(monkeypatch):
    tuples = [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=tuples, child_docs=["doc-a"])
    assert asyncio.run(reader.aload_data()) == ["doc-a"]
    assert len(fake.save_calls) == 1


# --- empty selection checkpoint save --------------------------------------

def test_load_data_empty_selection_saves_checkpoint(monkeypatch):
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=[], child_docs=[])
    assert reader.load_data() == []
    assert len(fake.save_calls) == 1


def test_iter_data_empty_selection_saves_checkpoint(monkeypatch):
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=[], child_docs=[])
    assert list(reader.iter_data()) == []
    assert len(fake.save_calls) == 1


def test_aload_data_empty_selection_saves_checkpoint(monkeypatch):
    reader, fake = _reader_for_load_tests(monkeypatch, tuples=[], child_docs=[])
    assert asyncio.run(reader.aload_data()) == []
    assert len(fake.save_calls) == 1


# --- recursive=False / input_files filtering ------------------------------

def test_filter_recursive_false_excludes_subdirectories(monkeypatch):
    reader = _bare_reader(
        _input_dir="/ifs/data",
        recursive=False,
    )
    assert reader._filter("/ifs/data/top.txt") is True
    assert reader._filter("/ifs/data/subdir/nested.txt") is False


def test_collect_files_recursive_false(monkeypatch):
    tuples = [
        (Path("/ifs/data/top.txt"), 10, 1, ["ENTRY_ADDED"]),
        (Path("/ifs/data/subdir/nested.txt"), 10, 2, ["ENTRY_ADDED"]),
    ]
    reader, fake = _reader_for_load_tests(
        monkeypatch,
        tuples=tuples,
        child_docs=["doc"],
        _input_dir="/ifs/data",
        recursive=False,
    )
    assert reader.load_data() == ["doc"]
    assert len(fake.save_calls) == 1


def test_filter_input_files_exact_match(monkeypatch):
    reader = _bare_reader(
        _explicit_files=["/ifs/data/allowed.txt"],
        _input_dir=None,
    )
    assert reader._filter("/ifs/data/allowed.txt") is True
    assert reader._filter("/ifs/data/other.txt") is False


def test_collect_files_input_files_exact_match(monkeypatch):
    tuples = [
        (Path("/ifs/data/allowed.txt"), 10, 1, ["ENTRY_ADDED"]),
        (Path("/ifs/data/extra.txt"), 10, 2, ["ENTRY_ADDED"]),
    ]
    captured = {}

    reader = _bare_reader(
        _force_scan=False,
        _explicit_files=["/ifs/data/allowed.txt"],
        _input_dir=None,
    )
    reader._PowerScaleSimpleDirectoryReader__pshelper = FakeHelper(tuples)

    def capture_child(files, meta_wrap):
        captured["files"] = list(files)
        return _FakeChildReader(["doc"])

    monkeypatch.setattr(reader, "_child_reader", capture_child)
    assert reader.load_data() == ["doc"]
    assert captured["files"] == ["/ifs/data/allowed.txt"]


# --- exclude_empty with fs.info() failure ---------------------------------

def test_filter_exclude_empty_info_failure_logs_warning(caplog):
    class ExplodingFS:
        def isfile(self, p):
            return True

        def info(self, p):
            raise OSError("cannot stat")

    reader = _bare_reader(exclude_empty=True, fs=ExplodingFS())
    assert reader._filter("/ifs/data/a.txt") is False
    assert "Could not determine size" in caplog.text


# --- dataset_name / input_files init / fs forwarding -----------------------

def test_init_accepts_dataset_name(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        dataset_name="my-dataset",
    )
    assert reader._dataset_name == "my-dataset"
    assert reader.input_dir is None
    assert reader.input_files == []


def test_init_dataset_name_rejects_other_scopes(monkeypatch):
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    with pytest.raises(ValueError, match="exactly one"):
        PowerScaleSimpleDirectoryReader(
            es_host_url="h", es_index_name="i", es_api_key="k",
            dataset_name="my-dataset",
            input_dir="/ifs/data",
        )


def test_init_input_files_missing_does_not_raise(monkeypatch):
    """Missing input_files do not cause construction to fail."""
    monkeypatch.setattr(SimpleDirectoryReader, "__init__", lambda self, **kw: None)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    reader = PowerScaleSimpleDirectoryReader(
        es_host_url="h", es_index_name="i", es_api_key="k",
        input_files=["/ifs/data/missing.txt"],
    )
    assert reader._explicit_files == ["/ifs/data/missing.txt"]


def test_load_data_forwards_fs_and_restores(monkeypatch):
    """A per-call fs override is passed to the child and restored."""
    class RecordingFS:
        def isdir(self, p):
            return True

        def info(self, p):
            return {"size": 10}

    custom_fs = RecordingFS()
    original_fs = get_default_fs()
    reader = _bare_reader(
        fs=original_fs,
        _force_scan=False,
    )
    reader._PowerScaleSimpleDirectoryReader__pshelper = FakeHelper(
        [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    )

    captured = {}

    def fake_child(files, meta_wrap):
        captured["fs"] = reader.fs
        return _FakeChildReader(["doc"])

    monkeypatch.setattr(reader, "_child_reader", fake_child)

    docs = reader.load_data(fs=custom_fs)
    assert docs == ["doc"]
    assert captured["fs"] is custom_fs
    assert reader.fs is original_fs


def test_aload_data_forwards_fs_and_restores(monkeypatch):
    """aload_data forwards the fs argument to load_data and restores it."""
    class RecordingFS:
        def isdir(self, p):
            return True

        def info(self, p):
            return {"size": 10}

    custom_fs = RecordingFS()
    original_fs = get_default_fs()
    reader = _bare_reader(
        fs=original_fs,
        _force_scan=False,
    )
    reader._PowerScaleSimpleDirectoryReader__pshelper = FakeHelper(
        [(Path("/ifs/data/a.txt"), 10, 1, ["ENTRY_ADDED"])]
    )

    captured = {}

    def fake_child(files, meta_wrap):
        captured["fs"] = reader.fs
        return _FakeChildReader(["doc"])

    monkeypatch.setattr(reader, "_child_reader", fake_child)

    docs = asyncio.run(reader.aload_data(fs=custom_fs))
    assert docs == ["doc"]
    assert captured["fs"] is custom_fs
    assert reader.fs is original_fs

