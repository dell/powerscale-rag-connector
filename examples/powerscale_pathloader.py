#!/usr/bin/env python3
"""
PowerScale PathLoader Example

A simple example demonstrating how to use the PowerScalePathLoader
to fetch file paths from PowerScale MetadataIQ.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple, List

from powerscale_rag_connector import PowerScalePathLoader
from dotenv import load_dotenv
import os

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()

ES_HOST_URL = os.getenv("ES_HOST_URL")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME")
ES_API_KEY = os.getenv("ES_API_KEY")
FOLDER_PATH = os.getenv("FOLDER_PATH")
FORCE_SCAN = os.getenv("FORCE_SCAN", "false").lower() == "true"
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def get_powerscale_files() -> Iterator[Tuple[Path, int, int, List[str]]]:
    """
    Get file metadata from PowerScale's MetadataIQ facility using PowerScalePathLoader.

    Returns:
        Iterator of tuples containing (Path, snapshot_id, lin, change_types)
    """
    logger.info("Getting files from PowerScale: path=%s", FOLDER_PATH)

    loader = PowerScalePathLoader(
        es_host_url=ES_HOST_URL,
        es_index_name=ES_INDEX_NAME,
        es_api_key=ES_API_KEY,
        folder_path=FOLDER_PATH,
        force_scan=FORCE_SCAN,
        verify_ssl=VERIFY_SSL,
        app_name="powerscale_pathloader_example",
        app_version=1,
    )

    # Return the full tuple from lazy_load
    for file_tuple in loader.lazy_load():
        filepath, snapshot, lin, change_types = file_tuple
        logger.info(
            "File found: %s (snapshot: %d, changes: %s)",
            filepath,
            snapshot,
            change_types,
        )
        yield file_tuple


def main():
    try:
        # Set debug logging if requested
        if DEBUG_MODE:
            logging.getLogger("powerscale_rag_connector").setLevel(logging.DEBUG)

        # Get files from PowerScale
        start_time = time.time()
        file_count = 0

        # Process each file
        for file_path, snapshot_id, lin, change_types in get_powerscale_files():
            file_count += 1
            logger.info(
                "Processing file %d: %s (snapshot: %d, changes: %s)",
                file_count,
                file_path,
                snapshot_id,
                change_types,
            )
            # In a real application, you would do something with the file here

        # Calculate and log statistics
        elapsed_time = time.time() - start_time
        files_per_second = file_count / elapsed_time if elapsed_time > 0 else 0

        logger.info("Processing complete: %d files processed", file_count)
        logger.info(
            "Time elapsed: %.2f seconds (%.2f files/sec)",
            elapsed_time,
            files_per_second,
        )

    except Exception as e:
        logger.error("Error running PowerScale PathLoader: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()