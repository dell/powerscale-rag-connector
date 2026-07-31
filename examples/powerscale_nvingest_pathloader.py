#!/usr/bin/env python3
"""
For the powerscale_nvingest_pathloader.py example, you'll need to install the NVIDIA Ingest client library.
This code is compatible with nv-ingest v2 API. This example shows how to use the PowerScalePathLoader
to monitor a folder and send new files to NVIDIA Ingest for processing.

To install the NVIDIA Ingest client library:

  pip install nv-ingest-client python-dotenv

For more detailed information about the NVIDIA Ingest client library, refer to the official NVIDIA NV-Ingest
client documentation at https://docs.nvidia.com/nemo/retriever/latest/extraction/nv-ingest-python-api/

Environment variables (.env file):
  NV_INGEST_ENDPOINT - NVIDIA Ingest endpoint
  NV_INGEST_PORT - NVIDIA Ingest port
  ES_HOST_URL - Elasticsearch host URL
  ES_INDEX_NAME - Elasticsearch index name
  ES_API_KEY - Elasticsearch API key
  FOLDER_PATH - PowerScale folder path to monitor
  FORCE_SCAN - Force full scan (default: False)
  VERIFY_SSL - Verify SSL certificates (default: True)
  DEBUG_MODE - Enable debug logging (default: False)
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client.client import RestClient
from powerscale_rag_connector import PowerScalePathLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "See examples/.env.example for configuration details."
        )
    return value


# Get environment variables at the top of the file
NV_INGEST_ENDPOINT = _require_env("NV_INGEST_ENDPOINT")
NV_INGEST_PORT = int(_require_env("NV_INGEST_PORT"))
ES_HOST_URL = _require_env("ES_HOST_URL")
ES_INDEX_NAME = _require_env("ES_INDEX_NAME")
ES_API_KEY = _require_env("ES_API_KEY")
FOLDER_PATH = _require_env("FOLDER_PATH")
FORCE_SCAN = os.getenv("FORCE_SCAN", "false").lower() == "true"
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def run_ingestor(file_path: Path):
    """
    Set up and run the ingestion process to send traffic to NVIDIA Ingest.

    Args:
        file_path: Path to the file to ingest
    """
    logger.debug("Ingesting file: %s", file_path)

    # Use v2 API pattern
    ingestor = Ingestor(
        message_client_allocator=RestClient,
        message_client_hostname=NV_INGEST_ENDPOINT,
        message_client_port=NV_INGEST_PORT,
        message_client_kwargs={"api_version": "v2"}
    )

    try:
        logger.info("Submitting job to nv-ingest for %s...", file_path)
        # Build pipeline and ingest
        ingestor.files([str(file_path)]) \
            .extract(extract_text=True, extract_tables=True, extract_charts=True, extract_images=False) \
            .pdf_split_config(pages_per_chunk=64) \
            .ingest(return_full_response=True)
        # In a real application, process the results here, e.g. store chunks in a
        # vector database (see powerscale_nvingest_langchain_doc_loader.py for a full example)
        logger.info("Ingestion of %s completed successfully.", file_path)
        return True
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", file_path, e)
        return False


def main():
    try:
        # Set debug logging if requested
        if DEBUG_MODE:
            logging.getLogger("powerscale_rag_connector").setLevel(logging.DEBUG)

        loader = PowerScalePathLoader(
            es_host_url=ES_HOST_URL,
            es_index_name=ES_INDEX_NAME,
            es_api_key=ES_API_KEY,
            folder_path=FOLDER_PATH,
            force_scan=FORCE_SCAN,
            verify_ssl=VERIFY_SSL,
            app_name="powerscale_nvingest",
            app_version=1,
        )

        # Process statistics
        start_time = time.time()
        file_count = 0
        success_count = 0
        error_count = 0

        # Process each file
        # IMPORTANT: Errors during processing will abort the loop and prevent checkpoint
        # advancement. The loader will retry failed files on the next run.
        for file_tuple in loader.lazy_load():
            filepath, _snapshot, _lin, _change_types = file_tuple
            file_count += 1
            logger.info("Processing file %d: %s", file_count, filepath)

            if run_ingestor(filepath):
                success_count += 1
            else:
                error_count += 1
                logger.error(
                    "Ingestion failed for %s - aborting to prevent checkpoint advancement",
                    filepath,
                )
                raise RuntimeError(f"NvIngest failed for {filepath}")

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
