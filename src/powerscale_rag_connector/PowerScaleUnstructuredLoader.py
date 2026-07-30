"""PowerScale LangChain UnstructuredLoader.

Identifies files that changed since the last checkpoint and extracts their
contents with ``langchain_unstructured.UnstructuredLoader``.  Each returned
``Document`` contains the extracted text in ``page_content`` and PowerScale
metadata (``source``, ``snapshot``, ``lin``, ``change_types``).
"""

import logging
from typing import Any, Iterator, Optional

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from .PowerScalePathLoader import PowerScalePathLoader

_logger = logging.getLogger(__name__)


class PowerScaleUnstructuredLoader(UnstructuredLoader):
    """PowerScale LangChain UnstructuredLoader.

    Subclasses ``langchain_unstructured.UnstructuredLoader`` so every partition
    option of the upstream loader remains available, while PowerScale MetadataIQ
    supplies the set of files to parse instead of a caller-provided path.

    ``file_path`` is reassigned for each discovered file before delegating to
    ``UnstructuredLoader.lazy_load()``.  A single instance is therefore not safe
    to iterate from two threads at once.
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
        raise_on_error: bool = False,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
        **unstructured_kwargs: Any,
    ) -> None:
        """Initialize the loader with a folder path or dataset name.

        PowerScale-specific parameters define the Elasticsearch scope and
        checkpointing behavior.  Any additional keyword arguments are forwarded
        verbatim to ``langchain_unstructured.UnstructuredLoader`` (for example
        ``partition_via_api``, ``api_key``, ``url``, ``post_processors``, or any
        ``unstructured`` partition option such as ``languages``).

        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the Elasticsearch index
            es_api_key: api_key for Elasticsearch in hashed (encoded) form
            folder_path: The starting folder path to read data files from; must begin with "/ifs"
            dataset_name: The name of the MetadataIQ dataset to load. Note: dataset_name and folder_path are mutually exclusive
            chunking_strategy: (Optional; inherited) Chunking strategy passed to the wrapped loader (e.g. "basic", "by_title"). Defaults to None, which returns each document element as a separate Document. To replicate the old mode="single" behaviour (one merged Document per file), use chunking_strategy="basic" with a large max_characters value set via the unstructured library.
            force_scan: Force scanning all data regardless of state. Defaults to False.
            raise_on_error: If True, re-raise parse errors after logging. If False
                (default), errors are logged and the generator continues with the next file; the
                checkpoint still advances after the run completes.
            verify_ssl: Whether to verify SSL certificates for Elasticsearch connection. Defaults to True.
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
            **unstructured_kwargs: (Optional; inherited) Extra arguments forwarded to ``UnstructuredLoader``.
        """
        if chunking_strategy is not None:
            unstructured_kwargs["chunking_strategy"] = chunking_strategy

        # file_path is supplied per file in lazy_load(); the Unstructured client and
        # partition options are built once here instead of once per file.
        super().__init__(file_path=None, **unstructured_kwargs)

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

    def lazy_load(self) -> Iterator[Document]:
        """Yield Documents for every file PowerScale reports as changed."""
        for file_path, snapshot, lin, change_types in self.path_loader.lazy_load():
            # Point the inherited loader at the current file.
            self.file_path = str(file_path)
            try:
                for doc in super().lazy_load():
                    # PowerScale fields are authoritative, so they are written last.
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["snapshot"] = snapshot
                    doc.metadata["lin"] = lin
                    doc.metadata["change_types"] = change_types

                    yield doc
            except Exception as e:
                _logger.error("Error loading file %s: %s", file_path, e)
                if self.__raise_on_error:
                    raise
