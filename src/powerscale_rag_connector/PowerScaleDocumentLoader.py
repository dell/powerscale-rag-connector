"""PowerScale LangChain DocumentLoader.

Identifies files that changed since the last checkpoint and yields one
``Document`` per file with an empty ``page_content`` and PowerScale metadata
(``source``, ``snapshot``, ``lin``, ``change_types``).  Use ``PowerScaleUnstructuredLoader``
to extract file contents.
"""

import logging
import os
from typing import Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .PowerScaleHelper import PowerScaleHelper

_logger = logging.getLogger(__name__)


class PowerScaleDocumentLoader(BaseLoader):
    """PowerScale LangChain DocumentLoader.

    Loads file metadata via LangChain's ``BaseLoader`` interface, leveraging
    PowerScale MetadataIQ to efficiently find files that have changed.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        force_scan: bool = False,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """Initialize the loader with a folder path or dataset name.

        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the Elasticsearch index
            es_api_key: api_key for Elasticsearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset_name: The name of the MetadataIQ dataset to load. Note: dataset_name and folder_path are mutually exclusive
            force_scan: Force scanning all data regardless of state
            verify_ssl: Whether to verify SSL certificates for Elasticsearch connection. Defaults to True.
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
        """
        self.__es_host_url = es_host_url
        self.__es_index_name = es_index_name
        self.__es_api_key = es_api_key
        self.__folder_path = folder_path
        self.__dataset_name = dataset_name
        self.__force_scan = force_scan
        self.__verify_ssl = verify_ssl
        self.__app_name = app_name
        self.__app_version = app_version
        self.__pshelper: Optional[PowerScaleHelper] = None  # defer initialization until first use via __helper property

    @property
    def __helper(self) -> PowerScaleHelper:
        if self.__pshelper is None:
            self.__pshelper = PowerScaleHelper(
                es_host_url=self.__es_host_url,
                es_index_name=self.__es_index_name,
                es_api_key=self.__es_api_key,
                folder_path=self.__folder_path,
                dataset_name=self.__dataset_name,
                verify_ssl=self.__verify_ssl,
                app_name=self.__app_name,
                app_version=self.__app_version,
            )
        return self.__pshelper

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load new files on current path using MetadataIQ metadata.

        Checkpoint advancement is deferred; call :meth:`save_checkpoint` after
        downstream ingestion succeeds.
        """
        if self.__force_scan:
            file_generator = self.__helper.get_directory_changes(snapshot_id=0, save_checkpoint=False)
        else:
            file_generator = self.__helper.get_directory_changes(save_checkpoint=False)

        for file, snapshot, lin, change_types in file_generator:
            metadata = {
                "source": str(file),
                "snapshot": snapshot,
                "lin": lin,
                "change_types": change_types
            }
            _logger.debug(
                "File found=%s (snapshot: %d, changes: %s)",
                metadata["source"],
                metadata["snapshot"],
                metadata["change_types"],
            )
            yield Document(page_content="", metadata=metadata)

    def save_checkpoint(self) -> None:
        """Persist the checkpoint after downstream ingestion has succeeded."""
        self.__helper.save_checkpoint()
