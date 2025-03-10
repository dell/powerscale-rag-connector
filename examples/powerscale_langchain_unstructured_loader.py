#!/usr/bin/env python3
"""
PowerScale LangChain Unstructured Loader Example

A simple example demonstrating how to use the PowerScaleUnstructuredLoader
to fetch and parse document content from PowerScale using LangChain's
UnstructuredFileLoader with PowerScale MetadataIQ.

Note that running the example requires either:
1. The unstructured-client package (pip install unstructured-client) along with 
   a valid UNSTRUCTURED_API_KEY environment variable; 
   (see https://python.langchain.com/docs/integrations/document_loaders/unstructured_file/)

2. The unstructured package (pip install "unstructured[all-docs]") along with a local install 
   of the required unstructured components 
   (see https://docs.unstructured.io/open-source/installation/full-installation)
"""

import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List, Dict, Any
from collections import defaultdict

from langchain_core.documents import Document

from powerscale_rag_connector import PowerScaleUnstructuredLoader
import config

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_parsed_documents() -> Iterator[Document]:
    """
    Get parsed Document objects from PowerScale using PowerScaleUnstructuredLoader.

    Returns:
        Iterator of LangChain Document objects with parsed content
    """
    logger.info(
        "Getting and parsing documents from PowerScale: path=%s", config.FOLDER_PATH
    )

    # Create the loader, using the 'elements' mode to get more granular document elements
    loader = PowerScaleUnstructuredLoader(
        es_host_url=config.ES_HOST_URL,
        es_index_name=config.ES_INDEX_NAME,
        es_api_key=config.ES_API_KEY,
        folder_path=config.FOLDER_PATH,
        force_scan=config.FORCE_SCAN,
        verify_ssl=config.VERIFY_SSL,
        # 'elements' mode splits the document into more granular chunks
        # Use 'single' mode if you want the entire document as a single chunk
        mode="elements",
        app_name="powerscale_unstructured_example",
        app_version=1,
    )

    # Return the parsed Document objects from the loader
    for document in loader.lazy_load():
        source = document.metadata.get("source", "Unknown")
        snapshot = document.metadata.get("snapshot", -1)
        change_types = document.metadata.get("change_types", [])
        element_type = document.metadata.get("category", "Unknown")

        # Log basic info about the document element (verbose mode)
        logger.debug(
            "Document element found: %s (type: %s, snapshot: %d, changes: %s)",
            source,
            element_type,
            snapshot,
            change_types,
        )

        # In a real application, you might want to filter by element type
        # For example, only use text elements or table elements
        yield document


def analyze_document_elements(documents: List[Document]) -> Dict[str, Any]:
    """
    Analyze the document elements to provide statistics and insights.

    Args:
        documents: List of Document objects from UnstructuredLoader

    Returns:
        Dictionary with analysis results
    """
    results = {
        "total_elements": len(documents),
        "elements_by_type": defaultdict(int),
        "elements_by_source": defaultdict(int),
        "avg_element_length": 0,
        "total_content_length": 0,
    }

    for doc in documents:
        # Count by element type
        element_type = doc.metadata.get("category", "Unknown")
        results["elements_by_type"][element_type] += 1

        # Count by source file
        source = doc.metadata.get("source", "Unknown")
        results["elements_by_source"][source] += 1

        # Track content length
        results["total_content_length"] += len(doc.page_content)

    # Calculate average element length
    if results["total_elements"] > 0:
        results["avg_element_length"] = (
            results["total_content_length"] / results["total_elements"]
        )

    return results


def main():
    try:
        # Set debug logging if requested
        if config.DEBUG_MODE:
            logger.setLevel(logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)

        # Get documents from PowerScale
        start_time = time.time()

        # Collect all documents to analyze them
        documents = list(get_parsed_documents())

        # Calculate and log statistics
        elapsed_time = time.time() - start_time
        docs_per_second = len(documents) / elapsed_time if elapsed_time > 0 else 0

        logger.info(
            "Processing complete: %d document elements processed", len(documents)
        )
        logger.info(
            "Time elapsed: %.2f seconds (%.2f elements/sec)",
            elapsed_time,
            docs_per_second,
        )

        # Analyze document elements
        analysis = analyze_document_elements(documents)

        # Print analysis results
        logger.info("Document element analysis:")
        logger.info("  Total elements: %d", analysis["total_elements"])
        logger.info(
            "  Average element length: %.2f characters", analysis["avg_element_length"]
        )

        logger.info("  Elements by type:")
        for element_type, count in analysis["elements_by_type"].items():
            logger.info("    - %s: %d", element_type, count)

        logger.info("  Elements by source file:")
        for source, count in analysis["elements_by_source"].items():
            source_path = Path(source).name  # Just show the filename, not the full path
            logger.info("    - %s: %d elements", source_path, count)

    except Exception as e:
        logger.error(
            "Error running PowerScale Unstructured Loader: %s", e, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
