"""Module providing a langchain-style PowerScale MetadataIQ Loader that returns Paths objects"""

import logging
from pathlib import Path

from typing import Iterator, Optional, Tuple, List

from .PowerScaleHelper import PowerScaleHelper


class PowerScalePathLoader:
    """LangChain-style Loader the uses Dell's PowerScale MetadataIQ feature to quickly
    identify files that have changed since the last run.
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
        """Initialize the loader with a file path.

        Args:
            es_host_url: fqdn or IP address of the ElasticSearch database
            es_index_name: name of the index
            es_api_key: api_key for ElasticSearch in hashed form
            folder_path: The starting folder path to read data from
            dataset: The name of the MetadataIQ dataset to load. Note: dataset and folder_path are mutually exclusive
            force_scan: Force scanning all files regardless of index state
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
        self.__pshelper: Optional[PowerScaleHelper] = None

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

    def lazy_load(self) -> Iterator[Tuple[Path, int, List[str]]]:
        """
        Lazy load only new files on current path using MetadataIQ metadata.
        Yields files one at a time via scroll API.

        Returns:
            Iterator yielding tuples containing:
            - Path: pathlib.Path object of the file
            - snapshot: MetadataIQ snapshot number
            - change_types: List of changes (e.g. ['ENTRY_ADDED'], ['ENTRY_DELETED'])
        """
        file_generator = None
        if self.__force_scan:
            # When force scanning, use get_directory_changes with snapshot_id=0
            file_generator = self.__helper.get_directory_changes(snapshot_id=0)
        else:
            # For normal operation, use get_directory_changes with default snapshot_id
            file_generator = self.__helper.get_directory_changes()

        for index, file_tuple in enumerate(file_generator):
            filepath, snapshot, change_types = file_tuple
            logging.debug(
                "File returned %d: %s (gen %d) changes: %s",
                index,
                filepath,
                snapshot,
                change_types,
            )
            yield file_tuple
