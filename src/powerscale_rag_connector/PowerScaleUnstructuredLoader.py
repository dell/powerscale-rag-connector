import logging
from typing import Iterator, Optional

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import UnstructuredFileLoader

from .PowerScalePathLoader import PowerScalePathLoader


class PowerScaleUnstructuredLoader(BaseLoader):
    """Loads files using UnstructuredFileLoader, leveraging PowerScale MetadataIQ to efficiently
    find files that have changed.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        mode: str = "single",
        force_scan: bool = False,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """Initialize with a file path.

        Args:
            es_host_url: URI of the ElasticSearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the ElasticSearch index
            es_api_key: api_key for ElasticSearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset: The name of the MetadataIQ dataset to load. Note: dataset and folder_path are mutually exclusive
            mode: The mode to use for UnstructuredFileLoader ("single" or "elements").
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
        self.__mode = mode
        self.__force_scan = force_scan
        self.__verify_ssl = verify_ssl
        self.__app_name = app_name
        self.__version = app_version

        self.path_loader = PowerScalePathLoader(
            es_host_url=self.__es_host_url,
            es_index_name=self.__es_index_name,
            es_api_key=self.__es_api_key,
            folder_path=self.__folder_path,
            dataset_name=self.__dataset_name,
            force_scan=self.__force_scan,
            verify_ssl=self.__verify_ssl,
            app_name=self.__app_name,
            app_version=self.__version,
        )

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from the file path."""
        for file_path, snapshot, change_types in self.path_loader.lazy_load():
            try:
                loader = UnstructuredFileLoader(
                    file_path=str(file_path), mode=self.__mode
                )
                for doc in loader.load():
                    # ensure the source is set correctly
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["snapshot"] = snapshot
                    doc.metadata["change_types"] = change_types
                    yield doc
            except Exception as e:
                logging.error("Error loading file %s: %s", file_path, e)
                continue
