import logging

from typing import Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .PowerScaleHelper import PowerScaleHelper


class PowerScaleDocumentLoader(BaseLoader):
    """LangChain Document Loader the uses Dell PowerScale's  MetadataIQ feature to "checkpoint" the loader and only
    read files that have changed between its last run.

    Applications requiring more flexibility (without strict LangChain DocumentLoader API compatibility)
    may prefer to use the PowerScalePathLoader.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: str | None = None,
        dataset_name: str | None = None,
        force_scan: bool = False,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """Initialize the loader with a file path or dataset name.

        Args:
            es_host_url: URI of the ElasticSearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the ElasticSearch index
            es_api_key: api_key for ElasticSearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset: The name of the MetadataIQ dataset to load. Note: dataset and folder_path are mutually exclusive
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
        self.__pshelper: Optional[PowerScaleHelper] = (
            None  # defer initialization until first use via __helper property
        )

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
        """Lazy load new files on current path using MetadataIQ metadata"""
        file_generator = None
        if self.__force_scan:
            file_generator = self.__helper.get_directory_changes(snapshot_id=0)
        else:
            file_generator = self.__helper.get_directory_changes()

        for file, snapshot, change_types in file_generator:
            metadata = {
                "source": str(file),
                "snapshot": snapshot,
                "change_types": change_types,
            }
            logging.debug(
                "File found=%s (snapshot: %d, changes: %s)",
                metadata["source"],
                metadata["snapshot"],
                metadata["change_types"],
            )
            try:
                yield Document(page_content="", metadata=metadata)
            except Exception as e:
                logging.error("Error generating Document for %s: %s", str(file), str(e))
                continue
