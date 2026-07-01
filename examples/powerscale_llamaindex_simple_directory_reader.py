#!/usr/bin/env python3

"""
PowerScale LlamaIndex Simple Directory Reader Example
"""

import logging
import time
import sys

from typing import Iterator

from llama_index.core.schema import Document
from powerscale_rag_connector import PowerScaleSimpleDirectoryReader
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
INPUT_DIR = os.getenv("INPUT_DIR")
FORCE_SCAN = os.getenv("FORCE_SCAN", "false").lower() == "true"
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def get_powerscale_documents() -> Iterator[Document]:
    logger.info(
        "Getting documents from PowerScale: path=%s", INPUT_DIR
    )

    reader = PowerScaleSimpleDirectoryReader(
        es_host_url=ES_HOST_URL,
        es_index_name=ES_INDEX_NAME,
        es_api_key=ES_API_KEY,
        input_dir=INPUT_DIR,  # NOTE: Use either input_dir OR input_files
        input_files=None,  # if used, only look for new instances of these files
        exclude=None,  # exclude is a list of files to NOT detect or load
        recursive=False,
        verify_ssl=VERIFY_SSL,
        force_scan=FORCE_SCAN,
        app_name="powerscale_llamaindex_simple_directory_reader_example",
        app_version=1,
    )

    for doc in reader.lazy_load_data():
        yield doc


def main():
    try:
        if DEBUG_MODE:
            logging.getLogger("powerscale_rag_connector").setLevel(logging.DEBUG)

        start = time.time()
        doc_count = 0

        for doc in get_powerscale_documents():
            doc_count += 1
            logger.info(
                "Document %d: %s (snapshot=%d, changes=%s)",
                doc_count,
                doc.metadata["source"],
                doc.metadata["snapshot"],
                doc.metadata["change_types"],
            )

        elapsed = time.time() - start
        rate = doc_count / elapsed if elapsed > 0 else 0

        logger.info("Done: processed %d docs", doc_count)
        logger.info("Elapsed: %.2f sec (%.2f docs/sec)", elapsed, rate)

    except Exception as e:
        logger.error("Error running reader: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()