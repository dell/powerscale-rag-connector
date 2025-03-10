# PowerScale RAG Connector Examples

This directory contains python examples demonstrating how to use PowerScale RAG Connector in a standalone configuration, in a LangChain application and with NVIDIA's NVIngest services.

## Setup and Configuration

1. **Copy the example configuration file**:

   ```bash
   cp config.py.example config.py
   ```

2. **Edit the configuration values** in `config.py` to match your environment:
   - PowerScale MetadataIQ connection settings:
     - Elasticsearch host URL
     - Elasticsearch Index name
     - Elasticsearch API key
     - SSL certificate verification for Elasticsearch (enable/disable)
   - File scan settings (folder path, incremental/full scanning))
   - Debug settings
   - _Optional:_ NVIDIA Ingest Service settings (endpoint and port) for testing with NVIngest

3. **Run an example**:

   ```bash
   export PYTHONPATH=../src:$PYTHONPATH

   python powerscale_pathloader_example.py
   # or
   python powerscale_langchain_doc_loader.py
   # or
   python powerscale_langchain_unstructured_loader.py
   # or
   python powerscale_nvingest_example.py
   ```

## Available Examples

- **powerscale_pathloader_example.py**: Basic example showing how to use PowerScalePathLoader to retrieve file paths and metadata from PowerScale MetadataIQ.

- **powerscale_langchain_doc_loader.py**: Demonstrates using PowerScaleDocumentLoader to create LangChain Document objects with metadata from PowerScale's MetadataIQ.

- **powerscale_langchain_unstructured_loader.py**: Shows how to use PowerScaleUnstructuredLoader to parse documents using LangChain's UnstructuredFileLoader, extracting structured elements from source documents.

- **powerscale_nvingest_example.py**: Demonstrates integration between PowerScale data and NVIDIA's NVIngest for text extraction, splitting, and embedding.

## Requirements

- PowerScale storage system with MetadataIQ configured
- Elasticsearch host with MetadataIQ index
- Python 3.8+
- Required Python packages (install via pip):
  - langchain_core
  - langchain_community
  - elasticsearch
  - _For unstructured loader:_ unstructured-client or local unstructured package and tools (see powerscale_langchain_unstructured_loader.py file header for more information)
  - _For NVIngest example:_ nv-ingest-client (see installation instructions below)

## Installing NVIDIA Ingest Client

For the `powerscale_nvingest_example.py` example, you'll need to install the NVIDIA Ingest client library. This code has been tested with nv-ingest v24.12.1.

For more detailed information about the NVIDIA Ingest client library, refer to the [official NVIDIA NV-Ingest client documentation](https://github.com/NVIDIA/nv-ingest/tree/main/client).

To install the NVIDIA Ingest client library:

```bash
git clone https://github.com/NVIDIA/nv-ingest.git
cd nv-ingest
git checkout tags/24.12.1
cd [POWERSCALE_REG_CONNECTOR_REPO_ROOT]/public/examples
pip install -r [NVIDIA_INGEST_REPO_ROOT]/client/requirements.txt
pip install [NVIDIA_INGEST_REPO_ROOT]/client
```
