# PowerScale RAG Connector Examples

This directory contains python examples demonstrating how to use PowerScale RAG Connector in a standalone configuration, in a LangChain application and with NVIDIA's NVIngest services.

## Setup and Configuration

1. **Copy the example configuration file**:

   ```bash
   cp .env.example .env
   ```

2. **Edit the configuration values** in `.env` to match your environment:
   - PowerScale MetadataIQ connection settings:
     - Elasticsearch host URL
     - Elasticsearch Index name
     - Elasticsearch API key
     - SSL certificate verification for Elasticsearch (enable/disable)
   - File scan settings (folder path, incremental/full scanning)
   - Debug settings
   - _Optional:_ NVIDIA Ingest Service settings (endpoint and port) for testing with NVIngest

3. **Establish PowerScale NFS Access**:
   
   On your Linux machine ensure NFS client library and RPC daemons are running. On Ubuntu you may need to run
   ```
   sudo apt-get install nfs-common
   ```
   Create a local directory where you will access the PowerScale cluster. This could be any directory.
   ```
   sudo mkdir -p /ifs
   ```
   Run the mount command to access the PowerScale over NFS
   ```
   sudo mount -t nfs <cluster>:/ifs /ifs
   ```

4. **Run an example**:

   ```bash
   export PYTHONPATH=../src:$PYTHONPATH

   python powerscale_pathloader.py
   # or
   python powerscale_langchain_doc_loader.py
   # or
   python powerscale_langchain_unstructured_loader.py
   # or
   python powerscale_llamaindex_simple_directory_reader.py
   # or
   python powerscale_llamaindex_unstructured_reader.py
   # or
   python powerscale_nvingest_langchain_doc_loader.py
   # or
   python powerscale_nvingest_llamaindex_simple_dir_reader.py
   # or
   python powerscale_nvingest_pathloader.py
   ```

## Available Examples

- **powerscale_pathloader.py**: Basic example showing how to use PowerScalePathLoader to retrieve file paths and metadata from PowerScale MetadataIQ.

- **powerscale_langchain_doc_loader.py**: Demonstrates using PowerScaleDocumentLoader to create LangChain Document objects with metadata from PowerScale's MetadataIQ.

- **powerscale_langchain_unstructured_loader.py**: Shows how to use PowerScaleUnstructuredLoader to parse documents using LangChain's UnstructuredLoader, extracting structured elements from source documents.

- **powerscale_llamaindex_simple_directory_reader.py**: Shows how to use PowerScaleSimpleDirectoryReader to parse documents using LlamaIndex's SimpleDirectoryReader to create LlamaIndex Document objects with metadata from PowerScale's MetadataIQ.

- **powerscale_llamaindex_unstructured_reader.py**: Shows how to use PowerScaleUnstructuredReader to parse documents using LlamaIndex's UnstructuredReader, extracting structured elements from source documents.

- **powerscale_nvingest_langchain_doc_loader.py**: End-to-end RAG pipeline using PowerScaleDocumentLoader — processes changed files through NvIngest v2, embeds chunks via NVIDIA NIM, and stores them in a LangChain ElasticsearchStore. Handles `ENTRY_MODIFIED` by deleting old chunks by `lin` before inserting new ones.

- **powerscale_nvingest_llamaindex_simple_dir_reader.py**: End-to-end RAG pipeline using PowerScaleSimpleDirectoryReader — same pipeline as above using LlamaIndex.

- **powerscale_nvingest_pathloader.py**: Standalone NvIngest example using PowerScalePathLoader to detect changed files and submit them to NvIngest v2 for text extraction.

## Requirements

- PowerScale storage system with MetadataIQ configured
- Elasticsearch host with MetadataIQ index
- Python 3.10+
- Python packages (install via `pip`):

  **Base (all examples)**
  - `elasticsearch`
  - `powerscale_rag_connector`

  **LangChain examples**
  - `langchain_core`
  - `langchain-unstructured` (for `powerscale_langchain_unstructured_loader.py`)

  **LlamaIndex examples**
  - `llama-index`

  **NVIDIA Ingest examples**
  - `nv-ingest-client` (see installation instructions below)

  **Unstructured loaders**
  - `unstructured-client` or a local `unstructured` package (see `powerscale_langchain_unstructured_loader.py` and `powerscale_llamaindex_unstructured_reader.py` headers for details)

  **RAG vectorstore examples**
  - LangChain (`powerscale_nvingest_langchain_doc_loader.py`): `langchain-elasticsearch` and `langchain-nvidia-ai-endpoints`
  - LlamaIndex (`powerscale_nvingest_llamaindex_simple_dir_reader.py`): `llama-index-vector-stores-elasticsearch` and `llama-index-embeddings-nvidia`
  - A running NVIDIA NIM embeddings container (see NIM setup below)

## Installing NVIDIA Ingest Client

The NvIngest examples use the v2 API. For more information refer to the
[official NV-Ingest documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/nv-ingest-python-api/).

To install:

```bash
pip install nv-ingest-client
```

> **Note on terminology:** This codebase refers to NVIDIA's document ingestion service as **NvIngest** throughout. Some NVIDIA documentation and older references use the name **NeMo Retriever** or **NeMo** interchangeably for the same service. These refer to the same product — see the [NeMo Retriever repository](https://github.com/NVIDIA/NeMo-Retriever) for more context.

## RAG Vectorstore Examples

Two end-to-end examples ingest changed files through NvIngest v2, embed them with NVIDIA NIM, and store the vectors in Elasticsearch:

- `powerscale_nvingest_langchain_doc_loader.py` (LangChain)
- `powerscale_nvingest_llamaindex_simple_dir_reader.py` (LlamaIndex)

### What is NVIDIA NIM?

NVIDIA NIM (Inference Microservice) is a pre-packaged AI model served in a Docker container with a built-in REST API. You start it with one command and immediately have an endpoint that accepts text and returns embedding vectors — lists of numbers that represent the semantic meaning of the text. These vectors are what get stored in the vectorstore and searched during RAG retrieval. NIM handles all model loading, GPU configuration, and serving automatically.

### Additional Requirements

The RAG vectorstore examples need NVIDIA Ingest and the packages below. A running NVIDIA NIM embeddings container is required for vector generation; refer to the [NVIDIA NIM documentation](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html) for deployment instructions.

For the LangChain example (`powerscale_nvingest_langchain_doc_loader.py`):

```bash
pip install langchain-elasticsearch langchain-nvidia-ai-endpoints
```

For the LlamaIndex example (`powerscale_nvingest_llamaindex_simple_dir_reader.py`):

```bash
pip install llama-index-vector-stores-elasticsearch llama-index-embeddings-nvidia
```

### Configuration

Set the following environment variables in addition to the standard MetadataIQ settings:

```bash
# NvIngest v2
NV_INGEST_ENDPOINT=<nvingest-host-or-ip>     # hostname or IP of the NvIngest service
NV_INGEST_PORT=7670

# Vectorstore (local Elasticsearch)
VECTORSTORE_ES_URL=http://localhost:9200
VECTORSTORE_INDEX=rag_vectors

# NVIDIA NIM embeddings
NVIDIA_EMBEDDING_URL=http://<nim-cluster-ip>:8000   # do NOT include /v1 — added automatically
NVIDIA_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_API_KEY=                                      # leave blank for local NIM

# Scan control
FORCE_SCAN=false    # set true to reprocess all files, ignoring checkpoint
```

### Running

Run the LangChain example:

```bash
python powerscale_nvingest_langchain_doc_loader.py
```

Run the LlamaIndex example:

```bash
python powerscale_nvingest_llamaindex_simple_dir_reader.py
```

Both examples use the environment variables configured above. Set `FORCE_SCAN=true` to reprocess all files from the beginning, ignoring the checkpoint.
