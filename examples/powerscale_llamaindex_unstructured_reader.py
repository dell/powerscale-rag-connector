#!/usr/bin/env python3
"""
PowerScale LlamaIndex Unstructured Reader Example

Demonstrates how to use PowerScaleUnstructuredReader to fetch and parse
document content from PowerScale using LlamaIndex's UnstructuredReader
with PowerScale MetadataIQ.

Requirements:
    pip install llama-index-core llama-index-readers-file unstructured
"""
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from llama_index.core import Document

from powerscale_rag_connector import PowerScaleUnstructuredReader

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()


def _require_env(name: str) -> str:
    """Return the value of env-var *name*, raising clearly if it is absent or empty."""
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Copy examples/.env.example to examples/.env and fill in the values."
        )
    return val


ES_HOST_URL = _require_env("ES_HOST_URL")
ES_INDEX_NAME = _require_env("ES_INDEX_NAME")
ES_API_KEY = _require_env("ES_API_KEY")
INPUT_DIR = _require_env("INPUT_DIR")
FORCE_SCAN = os.getenv("FORCE_SCAN", "false").lower() == "true"
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def get_parsed_documents() -> List[Document]:
    """
    Get parsed Document objects from PowerScale using PowerScaleUnstructuredReader.

    Returns:
        List of LlamaIndex Document objects with parsed content
    """
    logger.info("Fetching documents from PowerScale: path=%s", INPUT_DIR)

    # Create the reader, using 'elements' mode for granular chunks
    reader = PowerScaleUnstructuredReader(
        es_host_url=ES_HOST_URL,
        es_index_name=ES_INDEX_NAME,
        es_api_key=ES_API_KEY,
        folder_path=INPUT_DIR,
        force_scan=FORCE_SCAN,
        verify_ssl=VERIFY_SSL,
        mode="elements",  # 'single' for whole doc, 'elements' for granular chunks
        app_name="powerscale_unstructured_reader_example",
        app_version=1,
    )

    # Load all documents eagerly
    documents = reader.load_data()

    # Log details for debugging
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        snapshot = doc.metadata.get("snapshot", -1)
        change_types = doc.metadata.get("change_types", [])
        element_type = doc.metadata.get("category", "Unknown")

        logger.debug(
            "Document element: %s (type: %s, snapshot: %d, changes: %s)",
            source,
            element_type,
            snapshot,
            change_types,
        )

    return documents


def analyze_document_elements(documents: List[Document]) -> Dict[str, Any]:
    """
    Analyze the document elements to provide statistics and insights.
    """
    results = {
        "total_elements": len(documents),
        "elements_by_type": defaultdict(int),
        "elements_by_source": defaultdict(int),
        "avg_element_length": 0,
        "total_content_length": 0,
    }

    for doc in documents:
        element_type = doc.metadata.get("category", "Unknown")
        results["elements_by_type"][element_type] += 1

        source = doc.metadata.get("source", "Unknown")
        results["elements_by_source"][source] += 1

        results["total_content_length"] += len(doc.text)

    if results["total_elements"] > 0:
        results["avg_element_length"] = (
            results["total_content_length"] / results["total_elements"]
        )

    return results


def main():
    try:
        if DEBUG_MODE:
            logging.getLogger("powerscale_rag_connector").setLevel(logging.DEBUG)

        start_time = time.time()
        documents = get_parsed_documents()

        elapsed_time = time.time() - start_time
        docs_per_second = len(documents) / elapsed_time if elapsed_time > 0 else 0

        logger.info("Processing complete: %d elements", len(documents))
        logger.info("Time elapsed: %.2f sec (%.2f elements/sec)", elapsed_time, docs_per_second)

        analysis = analyze_document_elements(documents)

        logger.info("Document analysis:")
        logger.info("  Total elements: %d", analysis["total_elements"])
        logger.info("  Avg element length: %.2f chars", analysis["avg_element_length"])

        logger.info("  Elements by type:")
        for etype, count in analysis["elements_by_type"].items():
            logger.info("    - %s: %d", etype, count)

        logger.info("  Elements by source file:")
        for source, count in analysis["elements_by_source"].items():
            logger.info("    - %s: %d elements", Path(source).name, count)

    except Exception as e:
        logger.error("Error running PowerScale Unstructured Reader: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()