# Tests

Unit tests for the PowerScale RAG Connector. The suite runs fully offline: a
`FakeElasticsearch` (see `conftest.py`) stands in for a live cluster. By
default, `conftest.py` checks for optional framework packages (LangChain /
LlamaIndex) at the start of the test session and installs any missing ones with
`pip`. Use `--no-install-extras` to skip auto-installation and let the
framework-dependent tests be skipped if the packages are not present.

## Running

```bash
pytest                 # run all tests; auto-installs optional deps if missing
pytest -v              # verbose
pytest --no-install-extras   # do not auto-install optional packages
pytest tests/test_helper_changes.py    # a single module
```

To install dependencies manually instead:

```bash
pip install -e ".[test]"                 # core (Helper, PathLoader) only
pip install -e ".[test,langchain,llamaindex]"   # full coverage
```

## Layout

| File | Covers |
|------|--------|
| `conftest.py` | Shared `FakeElasticsearch`, `FakeHelper`, hit builders, `make_helper` fixture |
| `test_helper_init.py` | `PowerScaleHelper.__init__` scope validation & config |
| `test_helper_checkpoint.py` | `get_checkpoint`, `get_snapshot_id`, `get_saved_mtime`, `save_checkpoint`, `init_checkpoint_doc` |
| `test_helper_query.py` | `build_query`, `update_latest_snapid`, `es_search_paged` |
| `test_helper_changes.py` | `match_files_by_snapshot`, `get_directory_changes`, `get_new_files`, `get_deleted_files` |
| `test_loaders.py` | `PowerScalePathLoader`, `PowerScaleDocumentLoader` |
| `test_simple_directory_reader.py` | `PowerScaleSimpleDirectoryReader` filtering, metadata merge, lazy load |
| `test_unstructured.py` | `PowerScaleUnstructuredLoader`, `PowerScaleUnstructuredReader` |
| `test_package_init.py` | Public exports and lazy-import `__getattr__` |

## Notes

- Tests reach into name-mangled attributes (e.g. `helper._PowerScaleHelper__last_state`)
  to assert internal checkpoint state. This is deliberate for white-box coverage.
- Reader tests bypass the heavy `SimpleDirectoryReader.__init__` via `__new__`
  and only set the attributes each method needs.
