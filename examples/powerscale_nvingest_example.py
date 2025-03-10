# For the powerscale_nvingest_example.py example, you'll need to install the NVIDIA Ingest client library.
# This code has been developed and tested with nv-ingest v24.12.1.
#
# To install the NVIDIA Ingest client library:
#
#   git clone https://github.com/NVIDIA/nv-ingest.git
#   cd nv-ingest
#   git checkout tags/24.12.1
#   cd [POWERSCALE_REG_CONNECTOR_REPO_ROOT]/public/examples
#   pip install -r [NVIDIA_INGEST_REPO_ROOT]/client/requirements.txt
#   pip install [NVIDIA_INGEST_REPO_ROOT]/client
#
# For more detailed information about the NVIDIA Ingest client library, refer to the official NVIDIA NV-Ingest
# client documentation at https://github.com/NVIDIA/nv-ingest/tree/main/client
#

import logging
import sys
import time
from pathlib import Path
from typing import Iterator

from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.rest.rest_client import RestClient
from powerscale_rag_connector import PowerScalePathLoader
import config

# Configure the logger
logger = logging.getLogger(__name__)


def run_ingestor(file_path: Path):
    """
    Set up and run the ingestion process to send traffic to NVIDIA Ingest.

    Args:
        file_path: Path to the file to ingest
    """
    logger.debug("Ingesting file: %s", file_path)

    # Ensure the file exists
    if not file_path.exists():
        logger.error("File does not exist: %s", file_path)
        return False

    client = NvIngestClient(
        message_client_allocator=RestClient,
        message_client_port=config.NEMO_PORT,
        message_client_hostname=config.NEMO_ENDPOINT,
    )

    ingestor = (
        Ingestor(client=client)
        .files(str(file_path))
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
        )
        .split(
            split_by="word",
            split_length=300,
            split_overlap=10,
            max_character_length=5000,
            sentence_window_size=0,
        )
        .embed(text=True, tables=True)
    )

    try:
        # The result of the ingest() operation below will be a JSON dictionary of the
        # parsed document elements and metadata.  This is where you would add code to do what
        # you want with the chunks and embeddings.
        _ = ingestor.ingest()

        logger.debug("Ingestion of %s completed successfully.", file_path)
        return True
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", file_path, e)
        return False


def get_powerscale_files() -> Iterator[Path]:
    """
    Get files from PowerScale using PowerScalePathLoader.

    Returns:
        Iterator of Path objects
    """
    logger.debug("Getting files from PowerScale: path=%s", config.FOLDER_PATH)

    loader = PowerScalePathLoader(
        es_host_url=config.ES_HOST_URL,
        es_index_name=config.ES_INDEX_NAME,
        es_api_key=config.ES_API_KEY,
        folder_path=config.FOLDER_PATH,
        force_scan=config.FORCE_SCAN,
        verify_ssl=config.VERIFY_SSL,
        app_name="powerscale_nvingest",
        app_version=1,
    )

    # The loader returns tuples of (Path, snapshot_id, change_types), but we only need the Path
    for file_tuple in loader.lazy_load():
        filepath, snapshot, change_types = file_tuple
        logger.debug(
            "File found: %s (snapshot: %d, changes: %s)",
            filepath,
            snapshot,
            change_types,
        )
        yield filepath


def main():
    try:
        # Set debug logging if requested
        if config.DEBUG_MODE:
            logger.setLevel(logging.DEBUG)

        # Get files from PowerScale
        files_iterator = get_powerscale_files()

        # Process statistics
        start_time = time.time()
        file_count = 0
        success_count = 0
        error_count = 0

        # Process each file
        for file_path in files_iterator:
            file_count += 1
            logger.info("Processing file %d: %s", file_count, file_path)

            if run_ingestor(file_path):
                success_count += 1
            else:
                error_count += 1

        # Calculate and log statistics
        elapsed_time = time.time() - start_time
        files_per_second = file_count / elapsed_time if elapsed_time > 0 else 0

        logger.info("Processing complete: %d files processed", file_count)
        logger.info("Success: %d, Errors: %d", success_count, error_count)
        logger.info(
            "Time elapsed: %.2f seconds (%.2f files/sec)",
            elapsed_time,
            files_per_second,
        )

    except Exception as e:
        logger.error("Error running ingestion: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
