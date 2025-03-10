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
import config

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_powerscale_files() -> Iterator[Tuple[Path, int, List[str]]]:
    """
    Get file metadata from PowerScale's MetadatIQ facility using PowerScalePathLoader.

    Returns:
        Iterator of tuples containing (Path, snapshot_id, change_types)
    """
    logger.info("Getting files from PowerScale: path=%s", config.FOLDER_PATH)

    loader = PowerScalePathLoader(
        es_host_url=config.ES_HOST_URL,
        es_index_name=config.ES_INDEX_NAME,
        es_api_key=config.ES_API_KEY,
        folder_path=config.FOLDER_PATH,
        force_scan=config.FORCE_SCAN,
        verify_ssl=config.VERIFY_SSL,
        app_name="powerscale_pathloader_example",
        app_version=1,
    )

    # Return the full tuple from lazy_load
    for file_tuple in loader.lazy_load():
        filepath, snapshot, change_types = file_tuple
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
        if config.DEBUG_MODE:
            logger.setLevel(logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)

        # Get files from PowerScale
        start_time = time.time()
        file_count = 0

        # Process each file
        for file_path, snapshot_id, change_types in get_powerscale_files():
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
