# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `PowerScaleUnstructuredLoader` — LangChain loader wrapping `UnstructuredFileLoader` for structured element extraction.
- `PowerScaleHelper` — core Elasticsearch/MetadataIQ client with checkpoint management, incremental and full-scan modes, and `get_directory_changes()` / `get_new_files()` / `get_deleted_files()` helpers.
- LangChain-compatible `lazy_load()` interface across all loaders.

[Unreleased]: https://github.com/dell/powerscale-rag-connector/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/dell/powerscale-rag-connector/compare/v1.0.9...v2.0.0
[1.0.9]: https://github.com/dell/powerscale-rag-connector/compare/v1.0.0...v1.0.9
[1.0.0]: https://github.com/dell/powerscale-rag-connector/releases/tag/v1.0.0
