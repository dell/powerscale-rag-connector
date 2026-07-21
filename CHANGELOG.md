# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`fs` parameter in `PowerScaleSimpleDirectoryReader`**: accepts an optional `fsspec.AbstractFileSystem` for custom filesystem support, matching `SimpleDirectoryReader` API compatibility.
- **`raise_on_error` parameter in `PowerScaleUnstructuredLoader`, `PowerScaleUnstructuredReader`, and `PowerScaleSimpleDirectoryReader`**: controls whether parse errors are re-raised (True, default) or logged and skipped (False). Provides consistent, safe error handling across all loaders and readers.
- **`dataset_name` parameter in `PowerScaleSimpleDirectoryReader`**: supports MetadataIQ dataset definitions as a third selection scope alongside `input_dir` and `input_files`.

### Changed

- **`get_deleted_files()` now raises `NotImplementedError`**: MetadataIQ does not emit `ENTRY_DELETED` events in the current OneFS firmware version, so this helper no longer silently returns an empty iterator. This is a breaking API change; callers should catch `NotImplementedError` or stop using the method until MetadataIQ supports delete events.
- **`input_files` checkpoint keys are sorted**: `PowerScaleHelper` now stores the `input_files` list in canonical sorted order so reordering the same file list between runs does not create a duplicate checkpoint entry.
- **Environment validation in nvingest examples**: `NV_INGEST_ENDPOINT`, `NV_INGEST_PORT`, `FOLDER_PATH`, and `INPUT_DIR` are now validated with `_require_env()` on import, producing clear error messages when required variables are missing.
- **`requirements.txt` now uses `python-dotenv` instead of `dotenv`**: the `dotenv` package name does not provide the `dotenv.load_dotenv` import used by the examples; `python-dotenv` is the correct dependency.
- **Standardized `force_scan` parameter documentation**: all loaders and readers now use the same `force_scan: Force scanning all data regardless of state` docstring.
- **`PowerScaleUnstructuredLoader` per-file instantiation**: an `UnstructuredLoader` instance is created for each file as it is processed, because the underlying loader ties `file_path` to its construction. A future optimization could reuse a single instance if `langchain-unstructured` supports updating `file_path` after initialization.
- **`PowerScaleDocumentLoader` and `PowerScaleSimpleDirectoryReader` now skip missing files returned by MetadataIQ**: `PowerScaleDocumentLoader` checks `os.path.isfile` before yielding a metadata-only `Document`, and `PowerScaleSimpleDirectoryReader._filter` drops files that do not exist on the configured `fsspec` filesystem. `PowerScaleUnstructuredLoader` and `PowerScaleUnstructuredReader` rely on the underlying parser raising for missing files, controlled by `raise_on_error`.
- **Elasticsearch dependency pinned**: `pyproject.toml` and `requirements.txt` now require `elasticsearch>=8,<9` to ensure compatibility with the Elasticsearch 8.x API.
- **`langchain` extra updated**: replaced `langchain-community` with `langchain-unstructured` (and `langchain-core`); `PowerScaleUnstructuredLoader` migrated to the new `langchain-unstructured` `UnstructuredLoader` and the `chunking_strategy` parameter. This is a breaking change for callers that previously passed `mode=` to `PowerScaleUnstructuredLoader`.
- **`llamaindex` extra updated**: now includes `llama-index-readers-file` and `unstructured[pdf]`, which `PowerScaleUnstructuredReader` requires at runtime.
- **`test` optional extra and pytest configuration added**: `pyproject.toml` now defines `project.optional-dependencies.test = ["pytest>=7"]` and `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `pythonpath = ["src"]`.
- **README and examples README updated**: `README.md` now includes `PowerScaleUnstructuredLoader` usage and a deprecation note for `langchain-community`'s `UnstructuredFileLoader`; `examples/README.md` lists per-example package requirements and clearer LlamaIndex vectorstore instructions.

### Added

- **Unit test suite (`tests/`)**: offline pytest coverage for `PowerScaleHelper`, `PowerScalePathLoader`, `PowerScaleDocumentLoader`, `PowerScaleSimpleDirectoryReader`, the Unstructured loader/reader, and the package's lazy-import surface. Uses a `FakeElasticsearch`; framework-specific tests skip automatically when optional extras are not installed. Run with `pip install -e ".[test]"` then `pytest`.
- **Auto-install optional test dependencies**: `tests/conftest.py` now checks for missing optional framework packages (`langchain-core`, `langchain-unstructured`, `llama-index`, `llama-index-readers-file`, `unstructured`) at the start of a test session and installs them with `pip` unless `--no-install-extras` is passed. This lets the full suite run by default without manually installing extras first.
- **`unstructured[pdf]` added to `llamaindex` and `all` extras**: `llama-index-readers-file`'s `UnstructuredReader` requires the `unstructured[pdf]` extra at runtime for PDF parsing, so it is now included in the optional dependency sets.
- **`PowerScaleHelper.get_all_files()` restored**: convenience method that calls `match_files_by_snapshot(snapshot_id=0)` and returns all regular files in scope (snapshot > 0) regardless of checkpoint, without writing a checkpoint. Removed in a previous refactor; restored for backwards compatibility.
- **`dataset_name` test coverage** (`tests/test_helper_dataset.py`): 12 new tests covering dataset definition lookup, `NotFoundError` propagation, checkpoint root, `build_query` with dataset scope, `get_directory_changes` end-to-end with a dataset, `refresh_dataset()`, and `get_all_files()`.

### Fixed

- **`exclude_empty` now enforced in `PowerScaleSimpleDirectoryReader`**: zero-byte files are now correctly filtered when `exclude_empty=True`. Previously the parameter was stored but not checked during file filtering.
- **`PowerScaleSimpleDirectoryReader` now uses `self.fs` for all filesystem operations**: existence checks during initialization and filtering now use the configured `fsspec.AbstractFileSystem` instead of `os.path`, ensuring consistency with the base `SimpleDirectoryReader` behavior.
- **`refresh_dataset()` missing `None` guard restored**: calling `refresh_dataset()` on a folder-path-scoped helper (where `dataset_name=None`) now returns `{}` instead of crashing with `TypeError` when passing `None` to `Elasticsearch.get(id=None)`.
- **Lazy import returned the module instead of the class**: `from powerscale_rag_connector import PowerScaleDocumentLoader` (and the other lazily-loaded classes) resolved to the submodule rather than the class because `importlib.import_module` binds the same-named submodule onto the package. This made `PowerScaleDocumentLoader(...)` raise `TypeError: 'module' object is not callable`, breaking every framework example. `__getattr__` now caches the resolved class onto the package namespace.
- **Missing `import time` in `powerscale_nvingest_pathloader.py`**: the example referenced `time.time()` without importing `time`, raising `NameError` at runtime.
- **Deprecated `body=` in nvingest example deletes**: `delete_by_lin` in the LangChain and LlamaIndex nvingest examples now uses the `query=` kwarg, compatible with `elasticsearch>=8,<9`.
- **`get_snapshot_id()` state regression**: restored the `__last_state is None` guard so repeated calls after a missing checkpoint root no longer crash with `TypeError: 'NoneType' object is not subscriptable`.
- **`match_files_by_snapshot` force-scan / first-run query**: aligned with main-repo style by always using `all_files=False`, so the snapshot range filter is applied consistently. `force_scan` and first-run queries use `metadata.snapshots.s2 > snapshot_id` and `metadata.snapshots.s2 <= latest_snapshot_id`.
- **`save_checkpoint` brand-new-entry edge case**: `state_snapshot_id` now defaults to `-1` so a fresh checkpoint with `latest_snapshot_id = 0` is still written.
- **`update_latest_snapid` elasticsearch client compatibility**: removed the deprecated `body=` keyword in favor of direct `aggs=` / `size=` kwargs, compatible with `elasticsearch>=8,<9`.
- **`init_checkpoint_doc` placeholder rows**: removed hardcoded `__empty_*__` placeholder entries that could have leaked into checkpoint documents.
- **`PowerScaleSimpleDirectoryReader` base-attribute initialization**: `self.fs` and `self.file_metadata` are now initialized to the same defaults as `SimpleDirectoryReader` (`get_default_fs()` and `_DefaultFileMetadataFunc(self.fs)`) and `self.exclude` is set from the constructor parameter, fixing inherited base methods that rely on these attributes.
- **v1 checkpoint migration for `saved_mtime`**: `PowerScaleHelper` now tracks whether a checkpoint actually contains a `saved_mtime` value. Old v1 checkpoints that lack the field are no longer treated as `saved_mtime = 0`, preventing `ENTRY_MODIFIED` events from being incorrectly reclassified as `ENTRY_ADDED`.
- **`PowerScaleSimpleDirectoryReader` checkpoint and parsing ordering**: the checkpoint is now written only after the inner `SimpleDirectoryReader` has successfully parsed the selected files. A new `raise_on_error` parameter (default `True`) is propagated to the inner reader, so parse errors cause the run to fail before the checkpoint advances, preventing silent ingestion loss.
- **`input_files` scope restricted to exact requested files**: `PowerScaleSimpleDirectoryReader._filter` now verifies that paths returned by the Elasticsearch `match_phrase` query are actually in the requested `input_files` list, avoiding descendant-path false matches.
- **`recursive=False` now enforced**: `PowerScaleSimpleDirectoryReader._filter` drops files whose parent directory is not the configured `input_dir` when `recursive=False`.
- **`get_directory_changes` checkpoint control**: `get_directory_changes` now accepts an optional `save_checkpoint` parameter so callers can defer the checkpoint write until after downstream work completes.
- **`PowerScaleHelper` folder-path query no longer truncates large directories**: replaced `match_phrase_prefix` with `match_phrase` for `data.path` prefix matching. Elasticsearch's `match_phrase_prefix` query has a default `max_expansions` of 50, which silently dropped all but the first 50 files under any directory prefix. `match_phrase` matches the analyzed token sequence without expansion, so directories with more than 50 files are now fully scanned.
- **`dataset_name` now works with any valid Elasticsearch query**: `PowerScaleHelper.build_query` wraps non-`bool` dataset definitions in `bool.must` and normalizes `bool.must` to a list before appending the `data.file_type` and snapshot-range filters. Previously, dataset definitions with `bool.must` as a single dict or with a list of clauses produced invalid DSL and failed.

## [2.0.0] - 2026-07-11

### Added

- **`PowerScaleSimpleDirectoryReader`** — new LlamaIndex reader that wraps `SimpleDirectoryReader`, filtered to files changed since the last checkpoint via MetadataIQ.
- **`PowerScaleUnstructuredReader`** — new LlamaIndex reader that wraps `UnstructuredReader` for element-level document parsing, using MetadataIQ for file selection.
- **`lin` (logical inode number) in document metadata** — all loaders and readers now include `lin` in returned document metadata, enabling stable identification of files across renames and modifications.
- **btime-based change reclassification** — `ENTRY_MODIFIED` events where `btime > saved_mtime` (file was created and modified between runs) are now promoted to `ENTRY_ADDED`, preventing unnecessary delete-before-reindex on first-seen files.
- **NvIngest v2 examples**:
  - `powerscale_nvingest_langchain_doc_loader.py` — end-to-end RAG pipeline using `PowerScaleDocumentLoader`, NvIngest v2, NVIDIA NIM embeddings, and LangChain `ElasticsearchStore`.
  - `powerscale_nvingest_llamaindex_simple_dir_reader.py` — same pipeline using `PowerScaleSimpleDirectoryReader` and LlamaIndex.
  - `powerscale_nvingest_pathloader.py` — standalone NvIngest v2 example using `PowerScalePathLoader`.
- **`ENTRY_MODIFIED` vectorstore handling in NvIngest examples** — existing chunks are deleted by `lin` before re-indexing modified files, preventing stale content from appearing in RAG search results.
- **Optional dependency extras** in `pyproject.toml`:
  - `powerscale-rag-connector[langchain]` — installs LangChain dependencies.
  - `powerscale-rag-connector[llamaindex]` — installs LlamaIndex dependencies.
  - `powerscale-rag-connector[all]` — installs all optional dependencies.
- **`.env.example`** configuration file for examples, replacing the previous `config.py.example`.

### Changed

- **4-tuple return shape** — `PowerScaleHelper.get_directory_changes()` and `PowerScalePathLoader.lazy_load()` now yield `(Path, snapshot_id, lin, change_types)` instead of `(Path, snapshot_id, lin)`. Downstream loaders and examples updated accordingly.
- **Python minimum version** raised from `>=3.8` to `>=3.10`.
- **Package base dependencies** narrowed to `elasticsearch` only; `langchain-core` and `langchain-community` moved to the `langchain` optional extra.
- **Framework imports made lazy** in `__init__.py` — `PowerScaleDocumentLoader`, `PowerScaleUnstructuredLoader`, `PowerScaleSimpleDirectoryReader`, and `PowerScaleUnstructuredReader` are now imported on first use, allowing a base install (`elasticsearch` only) to succeed without LangChain or LlamaIndex present.

### Fixed

- `PowerScaleSimpleDirectoryReader` README snippet corrected to use `input_dir=` instead of `folder_path=`.
- Component links in README corrected to `./src/powerscale_rag_connector/` (were pointing to `./src/`).
- Examples `README.md` updated: setup instructions now reference `.env.example`/`.env`, script names corrected to match actual files, and `NVINGEST_PORT` env var corrected to `NV_INGEST_PORT`.
- Duplicate `LlamaIndex` entry removed from terminology table in README.

## [1.0.9] - 2025

### Fixed

- Header format fix in README.

## [1.0.0] - 2025

### Added

- Initial public release.
- `PowerScalePathLoader` — core module for identifying changed files via MetadataIQ.
- `PowerScaleDocumentLoader` — LangChain `BaseLoader` returning `Document` objects with MetadataIQ metadata.
- `PowerScaleUnstructuredLoader` — LangChain loader wrapping `UnstructuredLoader` for structured element extraction.
- `PowerScaleHelper` — core Elasticsearch/MetadataIQ client with checkpoint management, incremental and full-scan modes, and `get_directory_changes()` / `get_new_files()` / `get_deleted_files()` helpers.
- LangChain-compatible `lazy_load()` interface across all loaders.

[Unreleased]: https://github.com/dell/powerscale-rag-connector/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/dell/powerscale-rag-connector/compare/v1.0.9...v2.0.0
[1.0.9]: https://github.com/dell/powerscale-rag-connector/compare/v1.0.0...v1.0.9
[1.0.0]: https://github.com/dell/powerscale-rag-connector/releases/tag/v1.0.0
