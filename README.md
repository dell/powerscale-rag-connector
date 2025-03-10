# PowerScale RAG Connector

The PowerScale RAG Connector is an open-source Python library designed to enhance RAG application performance during data ingestion by skipping files that have already been processed. It leverages PowerScale's unique MetadataIQ capability to identify changed files within the OneFS filesystem and publish this information in an easily consumable format via ElasticSearch.

Developers can integrate the PowerScale RAG Connector directly within a LangChain RAG application as a supported document loader or use it independently as a generic Python class.

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
| Nvidia NIM Services | Part of Nvidia AI Enterprise, a set of microservices that can optional be used to efficiently chunk and embed files with GPU. The output of this data can be stored in a vector database for a RAG framework to use. |
| NV-Ingest | An Nvidia NIM microservice that will ingest complex office documents files with tables, and figures, and produce chunks and embedding to be stored in a vector database. |
| Chunking | The process of splitting the source file into smaller context aware pieces that can be searched and converted into vectors. Example: a chunk could be every paragraph within a large office document |
| Embedding | Turning a chunk of data into a vector where vector operations such as similarity, can be performed. |
| MetadataIQ | A new feature in PowerScale OneFS 9.10 that will periodically save filesystem metadata to an external database such as Elasticsearch |
| PowerScale RAG Connector | An open-source connector that can integrate with LangChain to improve data ingestion when data resides on PowerScale. |

## Installation

```bash
pip install powerscale-rag-connector
```

## Installing NVIDIA Ingest Client

To use the NVIDIA Ingest client with the PowerScale RAG Connector, you'll need to install the NVIDIA Ingest client library. This code has been tested with nv-ingest v24.12.1.

For more detailed information about the NVIDIA Ingest client library, refer to the [official NVIDIA NV-Ingest client documentation](https://github.com/NVIDIA/nv-ingest/tree/main/client).


## Usage

The PowerScale RAG Connector can be used in two ways:

1. As a LangChain document loader
2. As a standalone Python class

### Using as a LangChain Document Loader

```python
from langchain.document_loaders import PowerScaleDocumentLoader

# Initialize the loader
loader = PowerScaleDocumentLoader(
    es_host_url="http://elasticsearch:9200",
    es_index_name="metadataiq",
    es_api_key="your-api-key",
    folder_path="/ifs/data"
)

# Load documents
documents = loader.load()
```

### Using as a Standalone Path Loader

```python
from PowerScalePathLoader import PowerScalePathLoader

# Initialize the loader
loader = PowerScalePathLoader(
    es_host_url="http://elasticsearch:9200",
    es_index_name="metadataiq",
    es_api_key="your-api-key",
    folder_path="/ifs/data"
)

# Get changed files
changed_files = loader.lazy_load()
```

## Examples

Check out the [examples directory](./examples) for complete usage examples:

- [test Environment Configuration](./examples/config.py.example)
- [PowerScale NVIngest Integration](./examples/powerscale_nvingest_example.py)

## Components

The connector consists of several modules:

- [PowerScalePathLoader](./src/PowerScalePathLoader.py): Core module for identifying changed files
- [PowerScaleDocumentLoader](./src/PowerScaleDocumentLoader.py): Custom DocumentLoader for LangChain integration
- [PowerScaleUnstructuredLoader](./src/PowerScaleUnstructuredLoader.py): Custom Loader returning Documents processed by LangChain's UnstructuredFileLoader

## Requirements

- Python 3.8+
- Elasticsearch client
- PowerScale OneFS 9.10+ with MetadataIQ configured
- LangChain (optional, for LangChain integration)

## License

[MIT](https://github.com/dell/powerscale-rag-connector/blob/main/LICENSE)
