"""PowerScale LlamaIndex UnstructuredReader.

Identifies files that changed since the last checkpoint and extracts their
contents with ``llama_index.readers.file.UnstructuredReader``.  Each returned
``Document`` contains the extracted text in ``text`` and PowerScale metadata
(``source``, ``snapshot``, ``lin``, ``change_types``).
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader  # requires `llama-index-readers-file`

from .PowerScalePathLoader import PowerScalePathLoader

_logger = logging.getLogger(__name__)


class PowerScaleUnstructuredReader(UnstructuredReader):
    """PowerScale LlamaIndex UnstructuredReader.

    Subclasses ``llama_index.readers.file.UnstructuredReader`` so the upstream
    single-file contract is preserved: calling ``load_data(file=...)`` behaves
    exactly as the parent does.  Calling ``load_data()`` with no ``file``
    instead scans PowerScale MetadataIQ and parses every changed file.
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
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        allowed_metadata_types: Optional[Tuple] = None,
        excluded_metadata_keys: Optional[Set] = None,
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
            api_key: (optional, inherited) Unstructured.io API key. Defaults to None (local parsing).
            url: (optional, inherited) Unstructured.io API URL. Ignored unless `api_key` is set.
            allowed_metadata_types: (optional, inherited) Types permitted in document metadata.
            excluded_metadata_keys: (optional, inherited) Metadata keys to drop from documents.
        """
        super().__init__(
            api_key=api_key,
            url=url,
            allowed_metadata_types=allowed_metadata_types,
            excluded_metadata_keys=excluded_metadata_keys,
        )

        self.__split_documents = (mode == "elements")  # map mode string to LlamaIndex split_documents flag
        self.__languages = languages if languages is not None else ["en"]
        self.__raise_on_error = raise_on_error

        self.path_loader = PowerScalePathLoader(
            es_host_url=es_host_url,
            es_index_name=es_index_name,
            es_api_key=es_api_key,
            folder_path=folder_path,
            dataset_name=dataset_name,
            force_scan=force_scan,
            verify_ssl=verify_ssl,
            app_name=app_name,
            app_version=app_version,
        )

    def load_data(
        self,
        file: Optional[Path] = None,
        unstructured_kwargs: Optional[Dict] = None,
        document_kwargs: Optional[Dict] = None,
        extra_info: Optional[Dict] = None,
        split_documents: Optional[bool] = None,
        excluded_metadata_keys: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> List[Document]:
        """Load Documents, either from PowerScale or from a single explicit file.

        Keeps the inherited ``UnstructuredReader.load_data`` contract: when
        ``file`` is provided the call is delegated straight to the parent. When
        ``file`` is omitted, PowerScale MetadataIQ selects the files to parse.
        """
        if file is not None:
            return super().load_data(
                file=file,
                unstructured_kwargs=unstructured_kwargs,
                document_kwargs=document_kwargs,
                extra_info=extra_info,
                split_documents=(
                    self.__split_documents if split_documents is None else split_documents
                ),
                excluded_metadata_keys=excluded_metadata_keys,
            )
        return list(self.lazy_load_data())

    def lazy_load_data(self) -> Iterator[Document]:
        """
        Lazily yield LlamaIndex Documents from files discovered by PowerScalePathLoader.
        """
        for file_path, snapshot, lin, change_types in self.path_loader.lazy_load():
            try:
                # Suppress LlamaIndex doc_id deprecation warning emitted during load;
                # scoped here so it does not affect any other code in the process.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="'doc_id' is deprecated")
                    docs = super().load_data(
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

    def save_checkpoint(self) -> None:
        """Persist the checkpoint after downstream ingestion has succeeded.

        Note: ``load_data()`` and ``lazy_load_data()`` already commit the
        checkpoint when the generator is exhausted.  Calling this method
        explicitly is safe but typically unnecessary unless you break out of
        the generator early and still want to advance the checkpoint.
        """
        self.path_loader.save_checkpoint()
