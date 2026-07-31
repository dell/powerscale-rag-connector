# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`fs` parameter in `PowerScaleSimpleDirectoryReader`**: accepts an optional `fsspec.AbstractFileSystem` for custom filesystem support, matching `SimpleDirectoryReader` API compatibility.
- **`raise_on_error` parameter in `PowerScaleUnstructuredLoader`, `PowerScaleUnstructuredReader`, and `PowerScaleSimpleDirectoryReader`**: controls whether parse errors are re-raised (`True`) or logged and skipped (`False`, default). Provides consistent, safe error handling across all loaders and readers.
- **`dataset_name` parameter in `PowerScaleSimpleDirectoryReader`**: supports MetadataIQ dataset definitions as a third selection scope alongside `input_dir` and `input_files`.

- **Unit test suite (`tests/`)**: offline pytest coverage for `PowerScaleHelper`, `PowerScalePathLoader`, `PowerScaleDocumentLoader`, `PowerScaleSimpleDirectoryReader`, the Unstructured loader/reader, and the package's lazy-import surface. Uses a `FakeElasticsearch`; framework-specific tests skip automatically when optional extras are not installed. Run with `pip install -e ".[test]"` then `pytest`.
- **Opt-in auto-install of optional test dependencies**: `tests/conftest.py` checks for missing optional framework packages (`langchain-core`, `langchain-unstructured`, `llama-index`, `llama-index-readers-file`, `unstructured`) and installs them with `pip` only when `--install-extras` is passed. Without the flag, framework-dependent tests are skipped via `pytest.importorskip`.
- **`unstructured[pdf,doc,docx,xlsx,pptx,md]` added to `langchain`, `llamaindex`, and `all` extras**: `UnstructuredReader` and `UnstructuredLoader` require these `unstructured` extras at runtime for document parsing, so they are now included in the optional dependency sets.
- **`PowerScaleHelper.get_all_files()` restored**: convenience method that calls `match_files_by_snapshot(snapshot_id=0)` and returns all regular files in scope (snapshot > 0) regardless of checkpoint, without writing a checkpoint. Removed in a previous refactor; restored for backwards compatibility.
- **`dataset_name` test coverage** (`tests/test_helper_dataset.py`): 12 new tests covering dataset definition lookup, `NotFoundError` propagation, checkpoint root, `build_query` with dataset scope, `get_directory_changes` end-to-end with a dataset, `refresh_dataset()`, and `get_all_files()`.

### Changed

- **`PowerScaleUnstructuredLoader` now subclasses `langchain_unstructured.UnstructuredLoader`**: previously it subclassed only `BaseLoader` and wrapped `UnstructuredLoader` by composition. It now inherits from the concrete framework class, mirroring how `PowerScaleSimpleDirectoryReader` inherits from `SimpleDirectoryReader`. Any extra keyword arguments are forwarded to the parent, so `partition_via_api`, `api_key`, `url`, `post_processors`, `languages`, `strategy`, and every other `unstructured` partition option are now available.
- **`PowerScaleUnstructuredReader` now subclasses `llama_index.readers.file.UnstructuredReader`**: it previously subclassed only `BaseReader`. `load_data(file=...)` still behaves exactly as the upstream reader does; calling `load_data()` with no `file` runs the PowerScale MetadataIQ scan. `api_key`, `url`, `allowed_metadata_types`, and `excluded_metadata_keys` are now accepted and forwarded to the parent.
- **A single Unstructured instance is reused across files**: `UnstructuredLoader.lazy_load()` reads `self.file_path` at call time and keeps no other per-file state, so the connector retargets one instance per file instead of constructing a new loader (and a new `UnstructuredClient`) for every file.
- **Checkpoint auto-commits when generators are fully consumed (BREAKING)**: `get_directory_changes()` now defaults `save_checkpoint=True` and all loaders/readers auto-commit when their generators are fully consumed. `PowerScaleSimpleDirectoryReader` defers the checkpoint write until after parsing succeeds. Callers who previously relied on manual `save_checkpoint()` control must now explicitly pass `save_checkpoint=False` to defer.
- **`get_deleted_files()` now raises `NotImplementedError`**: MetadataIQ does not emit `ENTRY_DELETED` events in the current OneFS firmware version, so this helper no longer silently returns an empty iterator. This is a breaking API change; callers should catch `NotImplementedError` or stop using the method until MetadataIQ supports delete events.
- **`input_files` checkpoint keys are sorted**: `PowerScaleHelper` now stores the `input_files` list in canonical sorted order so reordering the same file list between runs does not create a duplicate checkpoint entry.
- **Environment validation in nvingest examples**: `NV_INGEST_ENDPOINT`, `NV_INGEST_PORT`, `FOLDER_PATH`, and `INPUT_DIR` are now validated with `_require_env()` on import, producing clear error messages when required variables are missing.
- **`requirements.txt` now uses `python-dotenv` instead of `dotenv`**: the `dotenv` package name does not provide the `dotenv.load_dotenv` import used by the examples; `python-dotenv` is the correct dependency.
- **Standardized `force_scan` parameter documentation**: all loaders and readers now use the same `force_scan: Force scanning all data regardless of state` docstring.
- **Removed local filesystem existence checks from `PowerScaleDocumentLoader`, `PowerScalePathLoader`, and `PowerScaleSimpleDirectoryReader`**: loaders and readers no longer pre-verify that files exist on `os.path` or `fsspec` before yielding them; files returned by MetadataIQ are passed directly to downstream parsers. Missing files now surface as parser errors, controlled by `raise_on_error`.
- **Tests updated to match the no-existence-check behavior**: `test_path_loader_yields_files_without_local_existence_check`, `test_filter_accepts_missing_file`, and the `fs` forwarding tests now assert that missing files are not pre-filtered and `isfile` is not called during filtering.
- **Elasticsearch dependency pinned**: `pyproject.toml` and `requirements.txt` now require `elasticsearch>=8,<9` to ensure compatibility with the Elasticsearch 8.x API.
- **`langchain` extra updated**: replaced `langchain-community` with `langchain-unstructured` (and `langchain-core`); `PowerScaleUnstructuredLoader` migrated to the new `langchain-unstructured` `UnstructuredLoader` and the `chunking_strategy` parameter. This is a breaking change for callers that previously passed `mode=` to `PowerScaleUnstructuredLoader`.
- **`llamaindex` extra updated**: now includes `llama-index-readers-file` and `unstructured[pdf,doc,docx,xlsx,pptx,md]`, which `PowerScaleUnstructuredReader` requires at runtime.
- **`test` optional extra and pytest configuration added**: `pyproject.toml` now defines `project.optional-dependencies.test = ["pytest>=7"]` and `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `pythonpath = ["src"]`.
- **README and examples README updated**: `README.md` now includes `PowerScaleUnstructuredLoader` usage and a deprecation note for `langchain-community`'s `UnstructuredFileLoader`; `examples/README.md` lists per-example package requirements and clearer LlamaIndex vectorstore instructions.

### Fixed

- **`PowerScaleHelper._in_scope()` post-filter for analyzed-field over-matches**: `get_directory_changes()` and `get_all_files()` now narrow Elasticsearch results to the requested scope. `folder_path` (`match_phrase_prefix`) no longer returns sibling directories (`/ifs/data/foo` no longer returns `/ifs/data/foobar`). `input_files` (`match_phrase`) no longer returns descendants or adjacent-token paths (`/ifs/data/report` does not return `/ifs/data/report/inner.txt` or `/ifs/data/report.bak`). `dataset_name` is intentionally not narrowed here.
- **`force_scan` on a fresh checkpoint now reclassifies `ENTRY_MODIFIED` as `ENTRY_ADDED`**: `is_first_run` required `snapshot_id < 0`, so a force scan (`snapshot_id=0`) with no saved checkpoint reported pre-existing files as `ENTRY_MODIFIED`. Callers then issued a delete-before-reindex by `lin` for vectors that had never been written. `is_first_run` now depends only on the absence of a checkpoint.
- **Removed the last local filesystem check from `PowerScaleSimpleDirectoryReader`**: `__init__` called `self.fs.isdir(input_dir)` and raised `ValueError("Directory does not exist")`, which prevented the reader from being constructed on hosts where `/ifs` is not mounted. Existence validation is the caller's responsibility; MetadataIQ owns file discovery.
- **`refresh_dataset()` now updates internal state**: previously returned the refreshed dataset but did not update `__dataset_doc`, causing subsequent queries to use stale data.
- **`fsspec` import for type annotation resolution**: `PowerScaleSimpleDirectoryReader` now imports `fsspec` so that `typing.get_type_hints()` can resolve the `fs` parameter annotation at runtime.
- **Removed unused `os` imports**: cleaned up `PowerScaleDocumentLoader` and `PowerScalePathLoader`.
- **`exclude_empty` now enforced in `PowerScaleSimpleDirectoryReader`**: zero-byte files are now correctly filtered when `exclude_empty=True`. Previously the parameter was stored but not checked during file filtering.
- **`PowerScaleSimpleDirectoryReader` now uses `self.fs` for filesystem operations**: size checks use the configured `fsspec.AbstractFileSystem` instead of `os.path`, ensuring consistency with the base `SimpleDirectoryReader` behavior.
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
- **`PowerScaleSimpleDirectoryReader` checkpoint and parsing ordering**: the checkpoint is now written only after the inner `SimpleDirectoryReader` has successfully parsed the selected files. A new `raise_on_error` parameter (default `False`) is propagated to the inner reader; when `True`, parse errors cause the run to fail before the checkpoint advances, preventing silent ingestion loss.
- **`input_files` scope restricted to exact requested files**: `PowerScaleSimpleDirectoryReader._filter` now verifies that paths returned by the Elasticsearch `match_phrase` query are actually in the requested `input_files` list, avoiding descendant-path false matches.
- **`recursive=False` now enforced**: `PowerScaleSimpleDirectoryReader._filter` drops files whose parent directory is not the configured `input_dir` when `recursive=False`.
- **`get_directory_changes` checkpoint control**: `get_directory_changes` now accepts an optional `save_checkpoint` parameter so callers can defer the checkpoint write until after downstream work completes.
- **`dataset_name` now works with any valid Elasticsearch query**: `PowerScaleHelper.build_query` wraps non-`bool` dataset definitions in `bool.must` and normalizes `bool.must` to a list before appending the `data.file_type` and snapshot-range filters. Previously, dataset definitions with `bool.must` as a single dict or with a list of clauses produced invalid DSL and failed.
- **`PowerScaleSimpleDirectoryReader._merge_metadata` no longer raises `KeyError`**: the child `SimpleDirectoryReader` may pass a path that does not exactly match the in-memory selected dict (e.g., after symlink resolution). The method now logs a warning and returns metadata without PowerScale fields instead of crashing.

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
