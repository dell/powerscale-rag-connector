#!/usr/bin/env python3
"""
PowerScale NvIngest RAG Vectorstore Example — LangChain Document Loader variant

This example uses PowerScaleDocumentLoader with NvIngest v2 and ElasticsearchStore.
It processes changed files from PowerScale, chunks them via NvIngest, generates embeddings
with NVIDIA NIM, and stores them in Elasticsearch.

PowerScaleDocumentLoader is a LangChain BaseLoader that wraps the same MetadataIQ
change detection. It yields Document objects where:
  - metadata["source"]       : file path on the PowerScale mount
  - metadata["lin"]          : PowerScale logical inode number
  - metadata["change_types"] : list of change type strings
  - metadata["snapshot"]     : MetadataIQ snapshot ID
  - page_content             : always "" — file content is extracted by NvIngest

Setup:
  pip install langchain langchain-elasticsearch langchain-nvidia-ai-endpoints
  pip install powerscale-rag-connector nv-ingest-client python-dotenv

Configuration (.env):
  ES_HOST_URL            - Elasticsearch URL for MetadataIQ, e.g. https://host:9200
  ES_INDEX_NAME          - MetadataIQ index name
  ES_API_KEY             - Elasticsearch API key for MetadataIQ
  FOLDER_PATH            - PowerScale folder path to monitor (must start with /ifs)
  VECTORSTORE_INDEX      - Elasticsearch index for vectorstore (default: rag_vectors)
  VECTORSTORE_ES_URL     - Elasticsearch URL for vectorstore (default: http://localhost:9200)
  NV_INGEST_ENDPOINT     - NvIngest service hostname or IP
  NV_INGEST_PORT         - NvIngest service port
  NVIDIA_EMBEDDING_URL   - NVIDIA NIM embeddings endpoint, e.g. http://<cluster-ip>:8000
  NVIDIA_API_KEY         - NVIDIA API key (optional for local NIM)
  NVIDIA_EMBEDDING_MODEL - Embedding model name, e.g. nvidia/llama-nemotron-embed-1b-v2
  FORCE_SCAN             - Scan all files ignoring checkpoint (default: False)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_elasticsearch import ElasticsearchStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client.client import RestClient
from powerscale_rag_connector import PowerScaleDocumentLoader

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()

ES_HOST_URL = os.getenv("ES_HOST_URL").rstrip("/")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME")
VECTORSTORE_INDEX = os.getenv("VECTORSTORE_INDEX", "rag_vectors")
VECTORSTORE_ES_URL = os.getenv("VECTORSTORE_ES_URL", "http://localhost:9200").rstrip("/")
ES_API_KEY = os.getenv("ES_API_KEY")
FOLDER_PATH = os.path.normpath(os.getenv("FOLDER_PATH"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"

NV_INGEST_ENDPOINT = os.getenv("NV_INGEST_ENDPOINT")
NV_INGEST_PORT = int(os.getenv("NV_INGEST_PORT"))
_nim_base = os.getenv("NVIDIA_EMBEDDING_URL", "").rstrip("/")
NIM_EMBED_URL = _nim_base if _nim_base.endswith("/v1") else f"{_nim_base}/v1" if _nim_base else ""
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "")
FORCE_SCAN = os.getenv("FORCE_SCAN", "false").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def get_vectorstore() -> ElasticsearchStore:
    embeddings = NVIDIAEmbeddings(
        model=NVIDIA_EMBEDDING_MODEL,
        base_url=NIM_EMBED_URL,
        api_key=NVIDIA_API_KEY if NVIDIA_API_KEY else None,
    )
    vectorstore = ElasticsearchStore(
        index_name=VECTORSTORE_INDEX,
        es_url=VECTORSTORE_ES_URL,
        embedding=embeddings,
    )
    logger.info("Vectorstore initialized: index=%s, es_url=%s", VECTORSTORE_INDEX, VECTORSTORE_ES_URL)
    return vectorstore


def run_nvingest(file_path: Path) -> List[Dict[str, Any]]:
    """Process a file through NvIngest v2 and return extracted text chunks."""
    if not file_path.exists():
        logger.error("File does not exist: %s", file_path)
        return []

    ingestor = Ingestor(
        message_client_allocator=RestClient,
        message_client_hostname=NV_INGEST_ENDPOINT,
        message_client_port=NV_INGEST_PORT,
        message_client_kwargs={"api_version": "v2"},
    )

    try:
        raw_results = (
            ingestor.files([str(file_path)])
            .extract(extract_text=True, extract_tables=True, extract_charts=True, extract_images=False)
            .pdf_split_config(pages_per_chunk=64)
            .ingest(return_full_response=True)
        )
        chunks = []
        for item in raw_results:
            if isinstance(item, dict):
                text = (
                    item.get("content")
                    or item.get("data")
                    or (item.get("metadata", {}).get("content") if isinstance(item.get("metadata"), dict) else None)
                    or ""
                )
                if text:
                    chunks.append({"text": str(text), "metadata": item.get("metadata", {})})
            elif hasattr(item, "content"):
                text = getattr(item, "content", "")
                if text:
                    chunks.append({"text": str(text), "metadata": getattr(item, "metadata", {})})
        logger.info("Ingestion of %s completed. %d chunks extracted.", file_path, len(chunks))
        return chunks
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", file_path, e)
        return []


def delete_by_lin(vectorstore: ElasticsearchStore, lin: int) -> int:
    try:
        response = vectorstore.client.delete_by_query(
            index=VECTORSTORE_INDEX,
            body={"query": {"term": {"metadata.lin": lin}}},
            refresh=True,
        )
        deleted_count = response.get("deleted", 0)
        logger.info("Deleted %d documents with lin=%d", deleted_count, lin)
        return deleted_count
    except Exception as e:
        logger.error("Failed to delete documents with lin=%d: %s", lin, e)
        return 0


def add_chunks_to_vectorstore(
    vectorstore: ElasticsearchStore,
    chunks: List[Dict[str, Any]],
    lin: int,
    source_path: str,
    change_types: List[str],
) -> int:
    try:
        texts = []
        metadatas = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            metadata["source"] = source_path
            metadata["lin"] = lin
            metadata["change_types"] = change_types
            texts.append(chunk.get("text", ""))
            metadatas.append(metadata)
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        logger.info("Added %d chunks to vectorstore for lin=%d", len(texts), lin)
        return len(texts)
    except Exception as e:
        logger.error("Failed to add chunks for lin=%d: %s", lin, e)
        return 0


def process_changed_files() -> None:
    vectorstore = get_vectorstore()

    loader = PowerScaleDocumentLoader(
        es_host_url=ES_HOST_URL,
        es_index_name=ES_INDEX_NAME,
        es_api_key=ES_API_KEY,
        folder_path=FOLDER_PATH,
        force_scan=FORCE_SCAN,
        verify_ssl=VERIFY_SSL,
        app_name="powerscale_nvingest_rag_doc",
        app_version=1,
    )

    file_count = success_count = error_count = 0

    for document in loader.lazy_load():
        file_count += 1
        filepath = Path(document.metadata.get("source", ""))
        lin = document.metadata.get("lin", 0)
        change_types = document.metadata.get("change_types", [])

        logger.info("Processing file %d: %s (lin=%d, change_types=%s)", file_count, filepath, lin, change_types)

        chunks = run_nvingest(filepath)
        if not chunks:
            logger.warning("No chunks produced for %s, skipping", filepath)
            error_count += 1
            continue

        if "ENTRY_MODIFIED" in change_types:
            deleted = delete_by_lin(vectorstore, lin)
            logger.info("Deleted %d old chunks for modified file (lin=%d)", deleted, lin)

        added = add_chunks_to_vectorstore(vectorstore, chunks, lin, str(filepath), change_types)
        if added > 0:
            success_count += 1
        else:
            error_count += 1

    logger.info("Processing complete: %d files, %d successful, %d errors", file_count, success_count, error_count)


if __name__ == "__main__":
    try:
        if DEBUG_MODE:
            logging.getLogger("powerscale_rag_connector").setLevel(logging.DEBUG)
        logger.info("Starting PowerScale NvIngest RAG processing (DocumentLoader)")
        logger.info("PowerScale path: %s", FOLDER_PATH)
        logger.info("MetadataIQ index: %s", ES_INDEX_NAME)
        logger.info("Vectorstore index: %s @ %s", VECTORSTORE_INDEX, VECTORSTORE_ES_URL)
        logger.info("NvIngest endpoint: %s:%s", NV_INGEST_ENDPOINT, NV_INGEST_PORT)
        logger.info("NIM embeddings endpoint: %s (model: %s)", NIM_EMBED_URL, NVIDIA_EMBEDDING_MODEL)
        process_changed_files()
    except Exception as e:
        logger.error("Error running processing: %s", e)
        sys.exit(1)
