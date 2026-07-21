"""PowerScale LangChain UnstructuredLoader.

Identifies files that changed since the last checkpoint and extracts their
contents with ``langchain_unstructured.UnstructuredLoader``.  Each returned
``Document`` contains the extracted text in ``page_content`` and PowerScale
metadata (``source``, ``snapshot``, ``lin``, ``change_types``).  Files missing
from the local filesystem are skipped with a warning.
"""

import logging
from typing import Iterator, Optional

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_unstructured import UnstructuredLoader
from .PowerScalePathLoader import PowerScalePathLoader

_logger = logging.getLogger(__name__)


class PowerScaleUnstructuredLoader(BaseLoader):
    """PowerScale LangChain UnstructuredLoader.

    Loads files via LangChain's ``UnstructuredLoader``, leveraging
    PowerScale MetadataIQ to efficiently find files that have changed.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        force_scan: bool = False,
        raise_on_error: bool = True,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """Initialize the loader with a folder path or dataset name.

        PowerScale-specific parameters define the Elasticsearch scope and
        checkpointing behavior.  ``chunking_strategy`` is optional and inherited
        from ``langchain_unstructured.UnstructuredLoader``; it controls how the
        underlying loader splits each file.  ``raise_on_error`` is a
        PowerScale-specific safety switch.

        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the Elasticsearch index
            es_api_key: api_key for Elasticsearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset_name: The name of the MetadataIQ dataset to load. Note: dataset_name and folder_path are mutually exclusive
            chunking_strategy: (Optional; inherited) Chunking strategy passed to the wrapped loader (e.g. "basic", "by_title"). Defaults to None, which returns each document element as a separate Document. To replicate the old mode="single" behaviour (one merged Document per file), use chunking_strategy="basic" with a large max_characters value set via the unstructured library.
            force_scan: Force scanning all data regardless of state. Defaults to False.
            raise_on_error: If True (default), re-raise parse errors after logging. If False, errors are logged and the generator continues with the next file; the checkpoint still advances after the run completes.
            verify_ssl: Whether to verify SSL certificates for Elasticsearch connection. Defaults to True.
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
        """
        self.__folder_path = folder_path
        self.__dataset_name = dataset_name
        self.__chunking_strategy = chunking_strategy
        self.__force_scan = force_scan
        self.__raise_on_error = raise_on_error
        self.__verify_ssl = verify_ssl
        self.__app_name = app_name
        self.__app_version = app_version

        self.path_loader = PowerScalePathLoader(
            es_host_url=es_host_url,
            es_index_name=es_index_name,
            es_api_key=es_api_key,
            folder_path=self.__folder_path,
            dataset_name=self.__dataset_name,
            force_scan=self.__force_scan,
            verify_ssl=self.__verify_ssl,
            app_name=self.__app_name,
            app_version=self.__app_version,
        )

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from the file path."""
        for file_path, snapshot, lin, change_types in self.path_loader.lazy_load():
            try:
                loader_kwargs = {}
                if self.__chunking_strategy is not None:
                    loader_kwargs["chunking_strategy"] = self.__chunking_strategy
                loader = UnstructuredLoader(file_path=str(file_path), **loader_kwargs)
                for doc in loader.lazy_load():
                    # ensure the source is set correctly
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["snapshot"] = snapshot
                    doc.metadata["lin"] = lin
                    doc.metadata["change_types"] = change_types

                    yield doc
            except Exception as e:
                _logger.error("Error loading file %s: %s", file_path, e)
                if self.__raise_on_error:
                    raise
        # Only commit the checkpoint once all files have been processed.
        self.path_loader.save_checkpoint()