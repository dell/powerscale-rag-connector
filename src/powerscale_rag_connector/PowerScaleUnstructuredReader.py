# PowerScaleUnstructuredReader.py

"""Module providing a LlamaIndex UnstructuredReader that uses Dell PowerScale MetadataIQ
to efficiently find files that have changed.
"""

import logging
import warnings
from pathlib import Path
from typing import Iterator, Optional, List

# Suppress Unstructured library deprecation warnings, will be updated when the Unstructured Library or LlamaIndex is updated
warnings.filterwarnings("ignore", message="'doc_id' is deprecated")

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import UnstructuredReader  # requires `llama-index-readers-file`

from .PowerScalePathLoader import PowerScalePathLoader


class PowerScaleUnstructuredReader(BaseReader):
    """
    Loads files via LlamaIndex's UnstructuredReader, leveraging PowerScale MetadataIQ
    to efficiently find files that have changed.
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
        """
        Args:
            es_host_url: URI of the ElasticSearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the ElasticSearch index
            es_api_key: api_key for ElasticSearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset_name: The name of the MetadataIQ dataset to load. Note: dataset_name and folder_path are mutually exclusive
            mode: Reader mode; "single" keeps file as one doc, "elements" yields element-level docs.
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
        self.__split_documents = (mode == "elements")  # map mode string to LlamaIndex split_documents flag
        self.__force_scan = force_scan
        self.__verify_ssl = verify_ssl
        self.__app_name = app_name
        self.__app_version = app_version

        # LlamaIndex Unstructured reader
        self._reader = UnstructuredReader()

        self.path_loader = PowerScalePathLoader(
            es_host_url=self.__es_host_url,
            es_index_name=self.__es_index_name,
            es_api_key=self.__es_api_key,
            folder_path=self.__folder_path,
            dataset_name=self.__dataset_name,
            force_scan=self.__force_scan,
            verify_ssl=self.__verify_ssl,
            app_name=self.__app_name,
            app_version=self.__app_version,
        )

    def load_data(self) -> List[Document]:
        return list(self.lazy_load_data())

    def lazy_load_data(self) -> Iterator[Document]:
        """
        Lazily yield LlamaIndex Documents from files discovered by PowerScalePathLoader.
        """
        for file_path, snapshot, lin, change_types in self.path_loader.lazy_load():
            try:
                docs = self._reader.load_data(
                    file=Path(str(file_path)),
                    split_documents=self.__split_documents,
                    languages=["en"],
                )
                for doc in docs:
                    metadata = (doc.metadata or {})
                    metadata["source"] = str(file_path)
                    metadata["snapshot"] = snapshot
                    metadata["lin"] = lin
                    metadata["change_types"] = change_types
                    doc.metadata = metadata
                    yield doc
            except Exception as e:
                logging.error("Error loading file %s (snapshot=%s, changes=%s): %s",
                              file_path, snapshot, change_types, e)
                continue

