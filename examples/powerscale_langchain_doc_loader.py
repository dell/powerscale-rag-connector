#!/usr/bin/env python3
"""
PowerScale LangChain Document Loader Example

A simple example demonstrating how to use the PowerScaleDocumentLoader
to fetch LangChain Document objects using metadata from PowerScale's MetadataIQ
facility.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List

from langchain_core.documents import Document

from powerscale_rag_connector import PowerScaleDocumentLoader
import config

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_powerscale_documents() -> Iterator[Document]:
    """
    Get Document objects from PowerScale using PowerScaleDocumentLoader.

    Returns:
        Iterator of LangChain Document objects
    """
    logger.info("Getting documents from PowerScale: path=%s", config.FOLDER_PATH)

    loader = PowerScaleDocumentLoader(
        es_host_url=config.ES_HOST_URL,
        es_index_name=config.ES_INDEX_NAME,
        es_api_key=config.ES_API_KEY,
        folder_path=config.FOLDER_PATH,
        force_scan=config.FORCE_SCAN,
        verify_ssl=config.VERIFY_SSL,
        app_name="powerscale_langchain_doc_loader",
        app_version=1,
    )

    # Return the Document objects from the loader
    for document in loader.lazy_load():
        yield document


def main():
    try:
        # Set debug logging if requested
        if config.DEBUG_MODE:
            logger.setLevel(logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)

        # Get documents from PowerScale
        start_time = time.time()
        doc_count = 0

        # Process each document
        for document in get_powerscale_documents():
            doc_count += 1
            logger.info(
                "Processing document %d: %s (snapshot: %d, changes: %s)",
                doc_count,
                document.metadata["source"],
                document.metadata["snapshot"],
                document.metadata["change_types"],
            )
            # In a real application, you would do something with the document here
            # For example, process the document content or add it to a vector store

        # Calculate and log statistics
        elapsed_time = time.time() - start_time
        docs_per_second = doc_count / elapsed_time if elapsed_time > 0 else 0

        logger.info("Processing complete: %d documents processed", doc_count)
        logger.info(
            "Time elapsed: %.2f seconds (%.2f docs/sec)",
            elapsed_time,
            docs_per_second,
        )

    except Exception as e:
        logger.error("Error running PowerScale Document Loader: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
