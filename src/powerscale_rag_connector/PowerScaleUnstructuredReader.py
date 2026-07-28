"""PowerScale LlamaIndex UnstructuredReader.

Identifies files that changed since the last checkpoint and extracts their
contents with ``llama_index.readers.file.UnstructuredReader``.  Each returned
``Document`` contains the extracted text in ``text`` and PowerScale metadata
(``source``, ``snapshot``, ``lin``, ``change_types``).
"""

import logging
import warnings
from pathlib import Path
from typing import Iterator, List, Optional

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import UnstructuredReader  # requires `llama-index-readers-file`

from .PowerScalePathLoader import PowerScalePathLoader

_logger = logging.getLogger(__name__)


class PowerScaleUnstructuredReader(BaseReader):
    """PowerScale LlamaIndex UnstructuredReader.

    Loads files via LlamaIndex's ``UnstructuredReader``, leveraging
    PowerScale MetadataIQ to efficiently find files that have changed.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        mode: str = "single",
        languages: Optional[List[str]] = None,
        force_scan: bool = False,
        raise_on_error: bool = False,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """
        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the Elasticsearch index
            es_api_key: api_key for Elasticsearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset_name: The name of the MetadataIQ dataset to load. Note: dataset_name and folder_path are mutually exclusive
            mode: Reader mode; "single" keeps file as one doc, "elements" yields element-level docs.
            languages: List of language codes for OCR hints (e.g. ["en"]). Defaults to ["en"].
            force_scan: Force scanning all data regardless of state
            raise_on_error: If True, re-raise parse errors after logging. If False
                (default), errors are logged and the generator continues with the next file;
                the checkpoint still advances after the run completes.
            verify_ssl: Whether to verify SSL certificates for Elasticsearch connection. Defaults to True.
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
        """
        self.__folder_path = folder_path
        self.__dataset_name = dataset_name
        self.__split_documents = (mode == "elements")  # map mode string to LlamaIndex split_documents flag
        self.__languages = languages if languages is not None else ["en"]
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

    def load_data(self, show_progress: bool = False) -> List[Document]:
        return list(self.lazy_load_data())

    def lazy_load_data(self) -> Iterator[Document]:
        """
        Lazily yield LlamaIndex Documents from files discovered by PowerScalePathLoader.
        """
        for file_path, snapshot, lin, change_types in self.path_loader.lazy_load():
            reader = UnstructuredReader()
            try:
                # Suppress LlamaIndex doc_id deprecation warning emitted during load;
                # scoped here so it does not affect any other code in the process.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="'doc_id' is deprecated")
                    docs = reader.load_data(
                        file=Path(file_path),
                        split_documents=self.__split_documents,
                        unstructured_kwargs={"languages": self.__languages},
                    )
                for doc in docs:
                    metadata = doc.metadata or {}
                    metadata["source"] = str(file_path)
                    metadata["snapshot"] = snapshot
                    metadata["lin"] = lin
                    metadata["change_types"] = change_types
                    doc.metadata = metadata
                    yield doc
            except Exception as e:
                _logger.error("Error loading file %s (snapshot=%s, changes=%s): %s",
                              file_path, snapshot, change_types, e)
                if self.__raise_on_error:
                    raise
        # Only commit the checkpoint once all files have been processed.
        self.path_loader.save_checkpoint()

