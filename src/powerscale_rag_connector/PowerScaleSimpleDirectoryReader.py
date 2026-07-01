import os
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from .PowerScaleHelper import PowerScaleHelper


class PowerScaleSimpleDirectoryReader(SimpleDirectoryReader):
    """LlamaIndex SimpleDirectoryReader using Dell PowerScale MetadataIQ.

    - `input_dir` mode: uses PowerScaleHelper.get_directory_changes().
    - `input_files` mode: uses PowerScaleHelper.get_directory_changes() with helper configured for input_files.
    - Exclusions, required extensions, hidden filtering, and local existence checks are applied
      before delegating to an inner SimpleDirectoryReader to read file contents.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        input_dir: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        recursive: bool = True,
        verify_ssl: bool = True,
        force_scan: bool = False,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
        file_extractor: Optional[Dict[str, SimpleDirectoryReader]] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        required_exts: Optional[Set[str]] = None,
        num_files_limit: Optional[int] = None,
        exclude_hidden: bool = False,
    ) -> None:
        """Initialize the reader with a PowerScale-backed selection scope.

        Args:
            es_host_url: URI of the ElasticSearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the ElasticSearch index
            es_api_key: api_key for ElasticSearch in hashed (encoded) form
            input_dir: Folder scope to scan; mutually exclusive with `input_files`
            input_files: Explicit list of files to check/load; mutually exclusive with `input_dir`
            exclude: Exact file paths to skip before reading
            recursive: Passed to the base SimpleDirectoryReader (folder mode semantics)
            verify_ssl: Whether to verify SSL certificates for the Elasticsearch client
            force_scan: If True, perform a full scan (snapshot_id=0); otherwise incremental (default)
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
            file_extractor: Optional mapping of extension → extractor for the inner reader
            file_metadata: Optional callback that returns per-file metadata dicts
            filename_as_id: Use filename as the document id
            encoding: Text encoding hint for the inner reader
            errors: Encoding error handling strategy ("ignore", "strict", etc.).
            required_exts: Only include files whose extension is in this set (e.g., {".pdf"}).
            num_files_limit: Maximum number of files to read this run.
            exclude_hidden: Skip files whose path contains any hidden component.
        """
        # Validate scope
        if (input_dir is None) == (input_files is None):
            raise ValueError("Select exactly one of `input_dir` or `input_files`.")
        if input_dir and not input_dir.startswith("/ifs"):
            raise ValueError("input_dir must start with '/ifs'")
        if input_dir and not os.path.isdir(input_dir):
            raise ValueError(f"Directory does not exist: {input_dir}")
        if input_files:
            for p in input_files:
                if not p.startswith("/ifs"):
                    raise ValueError(f"input_files path must start with '/ifs': {p}")
                if not os.path.isfile(p):
                    raise ValueError(f"File does not exist: {p}")

        super().__init__(
            input_dir=input_dir,
            input_files=input_files,
            exclude=None,
            recursive=recursive,
            required_exts=required_exts,
            num_files_limit=num_files_limit,
            file_extractor=file_extractor,
            file_metadata=file_metadata,
            filename_as_id=filename_as_id,
            errors=errors,
            encoding=encoding,
            exclude_hidden=exclude_hidden,
        )

        self._force_scan = force_scan
        self._exclude_exact: Set[str] = {os.path.normpath(p) for p in (exclude or [])}
        self._orig_cb = file_metadata
        self._explicit_files: Optional[List[str]] = (
            [os.path.normpath(p) for p in input_files] if input_files else None
        )
        self._input_dir: Optional[str] = os.path.normpath(input_dir) if input_dir else None

        self._es_host_url = es_host_url
        self._es_index_name = es_index_name
        self._es_api_key = es_api_key
        self._verify_ssl = verify_ssl
        self._app_name = app_name
        self._app_version = app_version

        self.__pshelper: Optional[PowerScaleHelper] = None

    @property
    def __helper(self) -> PowerScaleHelper:
        """Lazily initialize and return the PowerScale helper."""
        if self.__pshelper is None:
            if self._explicit_files:
                self.__pshelper = PowerScaleHelper(
                    es_host_url=self._es_host_url,
                    es_index_name=self._es_index_name,
                    es_api_key=self._es_api_key,
                    folder_path=None,
                    input_files=self._explicit_files,
                    dataset_name=None,
                    verify_ssl=self._verify_ssl,
                    app_name=self._app_name,
                    app_version=self._app_version,
                )
            else:
                self.__pshelper = PowerScaleHelper(
                    es_host_url=self._es_host_url,
                    es_index_name=self._es_index_name,
                    es_api_key=self._es_api_key,
                    folder_path=self._input_dir,
                    dataset_name=None,
                    verify_ssl=self._verify_ssl,
                    app_name=self._app_name,
                    app_version=self._app_version,
                )
        return self.__pshelper

    def _is_hidden(self, path_str: str) -> bool:
        """Return True if any path component is hidden (starts with a dot)."""
        p = Path(path_str)
        return any(part.startswith(".") for part in p.parts)

    def _filter(self, path_str: str) -> bool:
        """Apply local filters (exclude, existence, extension allow-list, hidden)."""
        p = os.path.normpath(path_str)

        if p in self._exclude_exact:
            return False

        if not os.path.isfile(p):
            return False

        if self.required_exts:
            ext = Path(p).suffix.lower()
            allowed = {
                e.lower() if e.startswith(".") else f".{e.lower()}"
                for e in self.required_exts
            }
            if ext not in allowed:
                return False

        if self.exclude_hidden and self._is_hidden(p):
            return False

        return True

    def _child_reader(self, files: List[str], meta_wrap: Callable[[str], Dict]) -> SimpleDirectoryReader:
        """Create an inner SimpleDirectoryReader for the selected files."""
        return SimpleDirectoryReader(
            input_files=files,
            input_dir=None,
            exclude=None,
            recursive=False,
            required_exts=self.required_exts,  # type: ignore
            num_files_limit=None,
            file_extractor=self.file_extractor,  # type: ignore
            file_metadata=meta_wrap,
            filename_as_id=self.filename_as_id,  # type: ignore
            errors=self.errors,  # type: ignore
            encoding=self.encoding,  # type: ignore
            exclude_hidden=self.exclude_hidden,  # type: ignore
        )

    def load_data(self) -> List[Document]:
        """Load all Documents."""
        return list(self.lazy_load_data())

    def lazy_load_data(self) -> Iterator[Document]:
        """Yield Documents whose source files were selected via PowerScale."""
        if self._force_scan:
            file_generator = self.__helper.get_directory_changes(snapshot_id=0)
        else:
            file_generator = self.__helper.get_directory_changes()

        selected: Dict[str, Tuple[int, int, List[str]]] = {}

        for file_path, snapshot, lin, changes in file_generator:
            p = os.path.normpath(str(file_path))
            if self._filter(p):
                selected[p] = (int(snapshot), int(lin), list(changes))

        ordered = sorted(selected.keys())
        if self.num_files_limit is not None:
            ordered = ordered[: self.num_files_limit]

        if not ordered:
            logging.debug("No files matched PowerScale selection or local filters.")
            return iter(())

        meta_wrap = partial(self._merge_metadata, selected=selected)
        child = self._child_reader(ordered, meta_wrap)
        for d in child.load_data():
            yield d

    def _merge_metadata(
        self,
        path_str: str,
        *,
        selected: Dict[str, Tuple[int, int, List[str]]],
    ) -> Dict:
        """Merge user metadata (if any) with PowerScale metadata."""
        user_meta: Dict = {}
        if self._orig_cb:
            try:
                user_meta = self._orig_cb(path_str) or {}
            except Exception as e:
                logging.debug("file_metadata failed for %s: %s", path_str, e)

        snapshot, lin, changes = selected[os.path.normpath(path_str)]
        meta = {"source": path_str, "snapshot": snapshot, "lin": lin, "change_types": changes}
        meta.update(user_meta)
        return meta