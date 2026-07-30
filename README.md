# PowerScale RAG Connector

The PowerScale RAG Connector is an open-source Python library designed to enhance RAG application performance during data ingestion by skipping files that have already been processed. It leverages PowerScale's unique MetadataIQ capability to identify changed files within the OneFS filesystem and publish this information in an easily consumable format via ElasticSearch.

Developers can integrate the PowerScale RAG Connector directly within a LangChain RAG application as a supported document loader, a LlamaIndex RAG application as a supported reader, or use it independently as a generic Python class.

## Workflow

![Workflow and integration of how the PowerScale RAG Connector integrates with the LangChain and NVIDIA AI Enterprise Software](powerscale-rag-connector-workflow.png)

*Figure 1: Workflow and integration of how the PowerScale RAG Connector integrates with the LangChain and NVIDIA AI Enterprise Software.*


## Audience

The intended audience for this document includes software developers, machine learning scientists, and AI developers who will utilize files from PowerScale in the development of a RAG application.

## Overview

This guide is divided into two sections: setting up the environment and using the connector. Note that system administration privileges are required for the initial configuration on PowerScale, which may need to be performed by PowerScale administrators.

## Terminology

| Term | Definition |
|------|------------|
| RAG | Retrieval Augmented Generation. A technique used to take an off the shelf large language model and provide the LLM context to data it has no knowledge of. |
| LangChain | LangChain is an open-source python and javascript framework used to help developers create RAG applications. |
| LlamaIndex | LlamaIndex is an open-source Python framework for building RAG applications. |
| Nvidia NIM Services | Part of Nvidia AI Enterprise, a set of microservices that can optional be used to efficiently chunk and embed files with GPU. The output of this data can be stored in a vector database for a RAG framework to use. |
| NV-Ingest | An Nvidia NIM microservice that will ingest complex office documents files with tables, and figures, and produce chunks and embedding to be stored in a vector database. |
| Chunking | The process of splitting the source file into smaller context aware pieces that can be searched and converted into vectors. Example: a chunk could be every paragraph within a large office document |
| Embedding | Turning a chunk of data into a vector where vector operations such as similarity, can be performed. |
| MetadataIQ | A new feature in PowerScale OneFS 9.10 that will periodically save filesystem metadata to an external database such as Elasticsearch |
| lin | Logical inode number. A unique and stable identifier for a file on PowerScale OneFS, used to track file versions across renames and modifications. |
| PowerScale RAG Connector | An open-source connector that integrates with LangChain or LlamaIndex to improve data ingestion when data resides on PowerScale. |

## Installation

### Basic installation
```bash
pip install powerscale-rag-connector
```

### Install with LangChain dependencies
```bash
pip install powerscale-rag-connector[langchain]
```

### Install with LlamaIndex dependencies
```bash
pip install powerscale-rag-connector[llamaindex]
```

### Full installation (LangChain + LlamaIndex)
```bash
pip install powerscale-rag-connector[all]
```

## Installing NVIDIA Ingest Client

The NvIngest examples use the v2 API. For more information refer to the [official NV-Ingest documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/nv-ingest-python-api/).

```bash
pip install nv-ingest-client
```


## Usage

The PowerScale RAG Connector can be used in three ways:

1. As a LangChain document loader
2. As a LlamaIndex reader
3. As a standalone Python class

### Using as a LangChain Document Loader

```python
from powerscale_rag_connector import PowerScaleDocumentLoader

loader = PowerScaleDocumentLoader(
    es_host_url="https://elasticsearch:9200",
    es_index_name="isi-metadataiq-index.cluster.guid",
    es_api_key="your-encoded-api-key",
    folder_path="/ifs/data"
)

for doc in loader.lazy_load():
    print(doc.metadata["source"], doc.metadata["change_types"])

# Persist checkpoint so the next run only sees new changes
loader.save_checkpoint()
```

Each returned `Document` includes `source`, `snapshot`, `lin`, and `change_types` in its metadata.

### Handling Modified Files

The `lin` field (logical inode number) is a stable file identifier on OneFS that persists across renames and modifications. Use it to remove stale chunks from your vectorstore before re-ingesting a modified file.

```python
for doc in loader.lazy_load():
    lin = doc.metadata["lin"]
    source = doc.metadata["source"]
    change_types = doc.metadata["change_types"]

    if "ENTRY_MODIFIED" in change_types:
        vectorstore.delete({"lin": lin})

    chunks = process_document(source)  # your chunking/embedding logic
    vectorstore.add(chunks, metadata={"lin": lin, "source": source})

# Persist checkpoint so the next run only sees new changes
loader.save_checkpoint()
```

> **Note on deleted files:** MetadataIQ does not emit `ENTRY_DELETED` events in the current OneFS firmware version. `get_deleted_files()` raises `NotImplementedError` accordingly. To handle deletes, you must track `lin` values in your own application layer and detect when a previously-seen `lin` stops appearing in results.

### Using as a LangChain Unstructured Loader

`PowerScaleUnstructuredLoader` **subclasses** `langchain-unstructured`'s `UnstructuredLoader`, so every partition option of the upstream loader stays available while PowerScale supplies the set of files to parse:

```python
from powerscale_rag_connector import PowerScaleUnstructuredLoader

loader = PowerScaleUnstructuredLoader(
    es_host_url="https://elasticsearch:9200",
    es_index_name="isi-metadataiq-index.cluster.guid",
    es_api_key="your-encoded-api-key",
    folder_path="/ifs/data",
    # chunking_strategy controls how unstructured partitions each file.
    # None (default): each document element is a separate Document object.
    # "basic": merge elements into larger contiguous chunks.
    # "by_title": chunk at section-title boundaries.
    chunking_strategy=None,
)

for doc in loader.lazy_load():
    print(doc.metadata["source"], doc.page_content[:80])

loader.save_checkpoint()
```

Any additional keyword arguments are forwarded to `UnstructuredLoader`, so upstream
options work unchanged:

```python
loader = PowerScaleUnstructuredLoader(
    es_host_url="https://elasticsearch:9200",
    es_index_name="isi-metadataiq-index.cluster.guid",
    es_api_key="your-encoded-api-key",
    folder_path="/ifs/data",
    chunking_strategy="by_title",
    strategy="hi_res",          # forwarded to unstructured
    languages=["en", "de"],     # forwarded to unstructured
    partition_via_api=False,    # forwarded to UnstructuredLoader
)
```

> **Deprecation note:** `langchain-community`'s `UnstructuredFileLoader` (the old loader
> that accepted a `mode=` parameter) is deprecated and has been replaced by
> `langchain-unstructured`'s `UnstructuredLoader`. `PowerScaleUnstructuredLoader` uses
> the new loader. The old `mode="single"` / `mode="elements"` parameter does not exist
> in the new API; use `chunking_strategy=` instead.

### Using as a LlamaIndex Reader

Two LlamaIndex readers are available. **`PowerScaleSimpleDirectoryReader`** wraps LlamaIndex's `SimpleDirectoryReader` filtered to changed files. It supports three mutually exclusive selection scopes: `input_dir`, `input_files`, or `dataset_name` (a MetadataIQ dataset definition in Elasticsearch):

```python
from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

reader = PowerScaleSimpleDirectoryReader(
    es_host_url="https://elasticsearch:9200",
    es_index_name="isi-metadataiq-index.cluster.guid",
    es_api_key="your-encoded-api-key",
    input_dir="/ifs/data",
    # Alternatively use input_files=[...] or dataset_name="my_dataset"
)

for doc in reader.lazy_load_data():
    print(doc.metadata["source"], doc.metadata["change_types"])

reader.save_checkpoint()
```

**`PowerScaleUnstructuredReader`** **subclasses** LlamaIndex's `UnstructuredReader` for element-level parsing:

```python
from powerscale_rag_connector import PowerScaleUnstructuredReader

reader = PowerScaleUnstructuredReader(
    es_host_url="https://elasticsearch:9200",
    es_index_name="isi-metadataiq-index.cluster.guid",
    es_api_key="your-encoded-api-key",
    folder_path="/ifs/data",
    mode="elements",   # 'single' keeps the whole file as one Document; 'elements' yields element-level Documents
    languages=["en"],  # optional OCR language hints
)

documents = reader.load_data()
reader.save_checkpoint()
```

Because it subclasses `UnstructuredReader`, the upstream single-file contract still
works and bypasses PowerScale entirely:

```python
# PowerScale-driven scan (no `file` argument)
documents = reader.load_data()

# Upstream UnstructuredReader behaviour: parse one explicit file, no MetadataIQ query
documents = reader.load_data(file=Path("/ifs/data/report.pdf"))
```

### Common reader/loader parameters

All loaders/readers that parse file content share a PowerScale-specific `raise_on_error` switch. The default is `False` for all of them:

- `PowerScaleUnstructuredLoader` / `PowerScaleUnstructuredReader`: `raise_on_error` (default `False`). If a file fails to parse, the error is logged and processing continues with the next file. The checkpoint advances at the end of the run, meaning failed files are not retried. Set it to `True` to re-raise the exception and stop ingestion; the checkpoint will not be advanced.
- `PowerScaleSimpleDirectoryReader`: `raise_on_error` (default `False`). Parse errors are logged and skipped; the checkpoint advances at the end of the run. Set it to `True` to re-raise parse errors and prevent checkpoint advancement, so the run can be retried.

When `raise_on_error` is `False` (default), failed files are logged and skipped, and the checkpoint advances at the end of the run, meaning failed files are not retried. When `raise_on_error` is `True`, the checkpoint is only advanced after all selected files have been successfully processed.

### Using as a Standalone Path Loader

```python
from powerscale_rag_connector import PowerScalePathLoader

# Initialize the loader
loader = PowerScalePathLoader(
    es_host_url="https://elasticsearch:9200",
    es_index_name="isi-metadataiq-index.cluster.guid",
    es_api_key="your-encoded-api-key",
    folder_path="/ifs/data"
)

# Get changed files
for path_info in loader.lazy_load():
    print(path_info)  # (Path, snapshot, lin, change_types)

# Persist checkpoint so the next run only sees new changes
loader.save_checkpoint()
```

## Examples

Check out the [examples directory](./examples) for complete usage examples:

- [Environment Configuration](./examples/.env.example)
- [PowerScale LangChain Document Loader](./examples/powerscale_langchain_doc_loader.py)
- [PowerScale LangChain Unstructured Loader](./examples/powerscale_langchain_unstructured_loader.py)
- [PowerScale LlamaIndex Simple Directory Reader](./examples/powerscale_llamaindex_simple_directory_reader.py)
- [PowerScale LlamaIndex Unstructured Reader](./examples/powerscale_llamaindex_unstructured_reader.py)
- [PowerScale Path Loader](./examples/powerscale_pathloader.py)
- [PowerScale NVIngest LangChain Document Loader](./examples/powerscale_nvingest_langchain_doc_loader.py)
- [PowerScale NVIngest LlamaIndex Simple Directory Reader](./examples/powerscale_nvingest_llamaindex_simple_dir_reader.py)
- [PowerScale NVIngest Path Loader](./examples/powerscale_nvingest_pathloader.py)

## Components

The connector consists of several modules:

- [PowerScalePathLoader](./src/powerscale_rag_connector/PowerScalePathLoader.py): Core module for identifying changed files
- [PowerScaleDocumentLoader](./src/powerscale_rag_connector/PowerScaleDocumentLoader.py): Custom DocumentLoader for LangChain integration
- [PowerScaleUnstructuredLoader](./src/powerscale_rag_connector/PowerScaleUnstructuredLoader.py): Custom Loader returning Documents processed by LangChain's UnstructuredLoader
- [PowerScaleSimpleDirectoryReader](./src/powerscale_rag_connector/PowerScaleSimpleDirectoryReader.py): Custom Simple Directory Reader for LlamaIndex integration
- [PowerScaleUnstructuredReader](./src/powerscale_rag_connector/PowerScaleUnstructuredReader.py): Custom Loader returning Documents processed by LlamaIndex's UnstructuredReader


## Requirements

- Python 3.10+
- Elasticsearch client
- PowerScale OneFS 9.10+ with MetadataIQ configured
- LangChain (optional, for LangChain integration)
- LlamaIndex (optional, for LlamaIndex integration)

## License

[MIT](https://github.com/dell/powerscale-rag-connector/blob/main/LICENSE)
