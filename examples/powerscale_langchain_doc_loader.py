#!/usr/bin/env python3
"""
PowerScale LangChain Document Loader Example

A simple example demonstrating how to use the PowerScaleDocumentLoader
to fetch LangChain Document objects using metadata from PowerScale's MetadataIQ
facility.
"""

import logging
import os
import sys
import time
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document

from powerscale_rag_connector import PowerScaleDocumentLoader

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
FOLDER_PATH = _require_env("FOLDER_PATH")
FORCE_SCAN = os.getenv("FORCE_SCAN", "false").lower() == "true"
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def get_powerscale_documents() -> Iterator[Document]:
    """
    Get Document objects from PowerScale using PowerScaleDocumentLoader.

    Returns:
        Iterator of LangChain Document objects
    """
    logger.info("Getting documents from PowerScale: path=%s", FOLDER_PATH)

    loader = PowerScaleDocumentLoader(
        es_host_url=ES_HOST_URL,
        es_index_name=ES_INDEX_NAME,
        es_api_key=ES_API_KEY,
        folder_path=FOLDER_PATH,
        force_scan=FORCE_SCAN,
        verify_ssl=VERIFY_SSL,
        app_name="powerscale_langchain_doc_loader",
        app_version=1,
    )

    # Return the Document objects from the loader
    for document in loader.lazy_load():
        yield document


def main():
    try:
        # Set debug logging if requested
        if DEBUG_MODE:
            logging.getLogger("powerscale_rag_connector").setLevel(logging.DEBUG)

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