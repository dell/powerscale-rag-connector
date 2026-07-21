from __future__ import annotations

import asyncio
import os
import logging
import fnmatch
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.file.base import BaseReader, get_default_fs, _DefaultFileMetadataFunc
from llama_index.core.schema import Document

from .PowerScaleHelper import PowerScaleHelper

_logger = logging.getLogger(__name__)


class PowerScaleSimpleDirectoryReader(SimpleDirectoryReader):
    """PowerScale LlamaIndex SimpleDirectoryReader.

    Loads files via LlamaIndex's ``SimpleDirectoryReader``, leveraging
    PowerScale MetadataIQ to efficiently find files that have changed.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        input_dir: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        recursive: bool = True,
        verify_ssl: bool = True,
        force_scan: bool = False,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
        file_extractor: Optional[Dict[str, BaseReader]] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        required_exts: Optional[List[str]] = None,
        exclude_hidden: bool = True,
        exclude_empty: bool = False,
        raise_on_error: bool = True,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Initialize the reader with a PowerScale-backed selection scope.

        Args marked ``(optional, inherited)`` are optional and come from
        LlamaIndex's ``SimpleDirectoryReader``.

        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the Elasticsearch index
            es_api_key: api_key for Elasticsearch in hashed (encoded) form
            input_dir: Folder scope to scan; mutually exclusive with `input_files` and `dataset_name`
            input_files: Explicit list of files to check/load; mutually exclusive with `input_dir` and `dataset_name`
            dataset_name: MetadataIQ dataset name scope; mutually exclusive with `input_dir` and `input_files`
            exclude: (optional, inherited) Glob patterns to skip before reading (e.g. ["*.tmp", "/ifs/data/**/temp.*"]). Defaults to None.
            recursive: (optional, inherited) Scan subdirectories when using `input_dir`. Defaults to True.
            verify_ssl: Whether to verify SSL certificates for Elasticsearch. Defaults to True.
            force_scan: Force scanning all data regardless of state. Defaults to False.
            app_name: Application name for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: Version number for the checkpoint document. Defaults to 1.
            file_extractor: (optional, inherited) Extension → extractor mapping. Defaults to None.
            file_metadata: (optional, inherited) Callback returning per-file metadata. Defaults to None.
            filename_as_id: (optional, inherited) Use filename as document id. Defaults to False.
            encoding: (optional, inherited) Text encoding hint. Defaults to "utf-8".
            errors: (optional, inherited) Encoding error handling strategy. Defaults to "ignore".
            required_exts: (optional, inherited) Only include files with these extensions (e.g. [".pdf"]). Defaults to None.
            exclude_hidden: (optional, inherited) Skip hidden path components. Defaults to True.
            exclude_empty: (optional, inherited) Skip zero-byte files. Defaults to False.
            raise_on_error: (optional, inherited) If True (default), the inner reader raises on parse errors and the checkpoint is not advanced, so the run can be retried. If False, errors are logged and skipped; the checkpoint still advances after the run completes.
            fs: (optional, inherited) File system to use. Defaults to local.
        """
        # Normalize paths so the MetadataIQ helper and the local filesystem checks use
        # the same canonical representation.
        norm_input_dir = os.path.normpath(input_dir) if input_dir is not None else None
        norm_input_files = (
            [os.path.normpath(p) for p in input_files] if input_files is not None else None
        )

        # Set the filesystem before validation to support existence checks.
        self.fs = fs or get_default_fs()

        # Validate scope
        if sum(x is not None for x in (norm_input_dir, norm_input_files, dataset_name)) != 1:
            raise ValueError("Select exactly one of `input_dir`, `input_files`, or `dataset_name`.")
        if norm_input_files == []:
            raise ValueError("input_files cannot be empty")
        if norm_input_dir and not norm_input_dir.startswith("/ifs"):
            raise ValueError("input_dir must start with '/ifs'")
        if norm_input_dir and not self.fs.isdir(norm_input_dir):
            raise ValueError(f"Directory does not exist: {norm_input_dir}")
        if norm_input_files:
            for p in norm_input_files:
                if not p.startswith("/ifs"):
                    raise ValueError(f"input_files path must start with '/ifs': {p}")

        # SimpleDirectoryReader.__init__ would walk the local filesystem in _add_files().
        # Initialize BaseReader directly to defer file discovery to MetadataIQ.
        super(SimpleDirectoryReader, self).__init__()

        self.errors = errors
        self.encoding = encoding
        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.exclude_empty = exclude_empty
        self.required_exts = required_exts
        self.raise_on_error = raise_on_error
        self.file_extractor = file_extractor or {}
        self.file_metadata = file_metadata or _DefaultFileMetadataFunc(self.fs)
        self.filename_as_id = filename_as_id

        # Initialize path attributes compatible with SimpleDirectoryReader.
        # input_dir and dataset_name modes leave input_files empty; the file list is
        # resolved from MetadataIQ at load time.
        self._dataset_name = dataset_name
        if norm_input_dir is not None:
            self.input_dir = Path(norm_input_dir)
            self.input_files: List[Path] = []
        elif norm_input_files is not None:
            self.input_dir = None
            self.input_files = [Path(p) for p in norm_input_files]
        else:
            self.input_dir = None
            self.input_files = []

        self._force_scan = force_scan
        self._exclude_exact: Set[str] = {os.path.normpath(p) for p in (exclude or [])}
        self._orig_cb = file_metadata
        self._explicit_files: Optional[List[str]] = norm_input_files
        self._input_dir: Optional[str] = norm_input_dir

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
            if self._dataset_name is not None:
                self.__pshelper = PowerScaleHelper(
                    es_host_url=self._es_host_url,
                    es_index_name=self._es_index_name,
                    es_api_key=self._es_api_key,
                    folder_path=None,
                    input_files=None,
                    dataset_name=self._dataset_name,
                    verify_ssl=self._verify_ssl,
                    app_name=self._app_name,
                    app_version=self._app_version,
                )
            elif self._explicit_files is not None:
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
                    input_files=None,
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

    @staticmethod
    def _match_exclude_pattern(path: str, pattern: str) -> bool:
        """Return True if ``path`` matches ``pattern`` with glob semantics.

        Supports ``**`` (recursive), ``*`` (single-path-component wildcards),
        and shell-style character ranges. Absolute patterns are anchored at the
        root; relative patterns can match any suffix of the path.
        """
        path_parts = Path(path).parts
        pattern_parts = Path(pattern).parts

        if pattern_parts and pattern_parts[0] == "/":
            if not path_parts or path_parts[0] != "/":
                return False
            return PowerScaleSimpleDirectoryReader._match_parts(
                path_parts[1:], pattern_parts[1:], start_anchored=True
            )

        return PowerScaleSimpleDirectoryReader._match_parts(
            path_parts, pattern_parts, start_anchored=False
        )

    @staticmethod
    def _match_parts(path_parts, pattern_parts, start_anchored):
        """Glob-match a split path against a split pattern.

        ``**`` matches zero or more path components; other wildcards are
        evaluated per-component using :func:`fnmatch.fnmatchcase`.
        """
        n = len(path_parts)
        m = len(pattern_parts)

        @lru_cache(maxsize=None)
        def match(i, j):
            if j == m:
                return i == n
            if i == n:
                return all(part == "**" for part in pattern_parts[j:])
            if pattern_parts[j] == "**":
                return match(i, j + 1) or (i < n and match(i + 1, j))
            if fnmatch.fnmatchcase(path_parts[i], pattern_parts[j]):
                return match(i + 1, j + 1)
            return False

        if start_anchored:
            return match(0, 0)
        return any(match(start, 0) for start in range(n + 1))

    def _filter(self, path_str: str) -> bool:
        """Apply local filters (exclude, existence, extension allow-list, hidden, empty, input_files, recursive)."""
        p = os.path.normpath(path_str)

        if self._exclude_exact and any(
            self._match_exclude_pattern(p, pattern) for pattern in self._exclude_exact
        ):
            return False

        if not self.fs.isfile(p):
            _logger.warning(
                "Skipping %s: file returned by MetadataIQ does not exist on the configured filesystem",
                p,
            )
            return False

        # input_files mode: the ES query uses match_phrase on an analyzed text field,
        # so a descendant path containing the same token sequence can be returned.
        # Restrict to the exact requested list.
        if self._explicit_files is not None and p not in self._explicit_files:
            return False

        # recursive=False: only keep files directly under the configured input_dir
        if self._input_dir is not None and not self.recursive:
            parent = os.path.dirname(p)
            if os.path.normpath(parent) != os.path.normpath(self._input_dir):
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

        if self.exclude_empty:
            try:
                size = self.fs.info(p).get("size", 0)
                if size == 0:
                    return False
            except OSError as e:
                _logger.warning(
                    "Could not determine size of %s for exclude_empty filter: %s",
                    p,
                    e,
                )
                return False

        return True

    def _child_reader(self, files: List[str], meta_wrap: Callable[[str], Dict]) -> SimpleDirectoryReader:
        """Create an inner SimpleDirectoryReader for the selected files."""
        return SimpleDirectoryReader(
            input_files=files,
            input_dir=None,
            exclude=None,
            recursive=False,
            required_exts=self.required_exts,
            file_extractor=self.file_extractor,
            file_metadata=meta_wrap,
            filename_as_id=self.filename_as_id,
            errors=self.errors,
            encoding=self.encoding,
            exclude_hidden=self.exclude_hidden,
            raise_on_error=self.raise_on_error,
            fs=self.fs,
        )

    def _collect_files(
        self
    ) -> Dict[str, Tuple[int, int, List[str]]]:
        """Use PowerScale to get the filtered file set for this run.

        The checkpoint is not saved here; callers must save the checkpoint after
        files are successfully parsed.
        """
        if self._force_scan:
            file_generator = self.__helper.get_directory_changes(snapshot_id=0, save_checkpoint=False)
        else:
            file_generator = self.__helper.get_directory_changes(save_checkpoint=False)

        selected: Dict[str, Tuple[int, int, List[str]]] = {}

        for file_path, snapshot, lin, changes in file_generator:
            p = os.path.normpath(str(file_path))
            if self._filter(p):
                selected[p] = (int(snapshot), int(lin), list(changes))

        return selected

    def _reader_for(self, files: List[str], selected: Dict[str, Tuple[int, int, List[str]]]):
        """Build the inner SimpleDirectoryReader that loads the selected files."""
        meta_wrap = partial(self._merge_metadata, selected=selected)
        return self._child_reader(files, meta_wrap)

    def load_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        fs: Optional[Any] = None,
    ) -> List[Document]:
        """Load all Documents.

        Mirrors the signature of ``llama_index.core.SimpleDirectoryReader.load_data``
        so callers can pass ``show_progress``, ``num_workers``, and ``fs``.
        """
        original_fs = self.fs
        if fs is not None:
            self.fs = fs
        try:
            selected = self._collect_files()
            if not selected:
                _logger.debug("No files matched PowerScale selection or local filters.")
                # Advance the checkpoint on an empty, fully-completed scan to avoid re-scanning.
                self.__helper.save_checkpoint()
                return []

            ordered = sorted(selected.keys())
            child = self._reader_for(ordered, selected)
            # The metadata partial captures this instance (including the ES client), so it
            # is not picklable.  Force sequential loading to avoid a multiprocessing failure.
            if isinstance(num_workers, int) and num_workers > 1:
                _logger.warning(
                    "num_workers > 1 is not supported by PowerScaleSimpleDirectoryReader; "
                    "using sequential load."
                )
                num_workers = None
            # fs is already passed to the child constructor via self.fs in _child_reader().
            documents = child.load_data(
                show_progress=show_progress, num_workers=num_workers
            )
            # Advance the checkpoint only after all selected files are parsed successfully.
            self.__helper.save_checkpoint()
            return documents
        finally:
            self.fs = original_fs

    def iter_data(
        self, show_progress: bool = False
    ) -> Generator[List[Document], Any, Any]:
        """Load data iteratively from the PowerScale selection.

        Yields one list of :class:`Document` objects per selected file. The
        checkpoint is advanced only when the generator is fully consumed.
        """
        selected = self._collect_files()
        if not selected:
            _logger.debug("No files matched PowerScale selection or local filters.")
            # Advance the checkpoint on an empty, fully-completed scan to avoid re-scanning.
            self.__helper.save_checkpoint()
            return

        ordered = sorted(selected.keys())
        child = self._reader_for(ordered, selected)
        fully_consumed = False
        try:
            for docs in child.iter_data(show_progress=show_progress):
                yield docs
            fully_consumed = True
        finally:
            # Only advance the checkpoint if every selected file was successfully
            # yielded and the caller did not break early.
            if fully_consumed:
                self.__helper.save_checkpoint()

    def lazy_load_data(
        self, show_progress: bool = False, **kwargs: Any
    ) -> Iterator[Document]:
        """Yield Documents one file at a time."""
        for docs in self.iter_data(show_progress=show_progress):
            yield from docs

    async def aload_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        fs: Optional[Any] = None,
    ) -> List[Document]:
        """Asynchronously load all Documents.

        Mirrors the signature of ``llama_index.core.SimpleDirectoryReader.aload_data``.
        """
        return await asyncio.to_thread(
            self.load_data,
            show_progress=show_progress,
            num_workers=num_workers,
            fs=fs,
        )

    def list_resources(self, *args: Any, **kwargs: Any) -> List[str]:
        """List the file paths currently selected by PowerScale.

        **Important**: This method performs a PowerScale MetadataIQ scan via
        Elasticsearch and returns all matching files.
        Unlike the base ``SimpleDirectoryReader.list_resources()`` which returns
        the static ``input_files`` list, this method reflects the current PowerScale
        selection and may be I/O-intensive for large datasets.

        Returns:
            List of absolute file paths matching the PowerScale scope.
        """
        selected = self._collect_files()
        return sorted(selected.keys())

    def _merge_metadata(
        self,
        path_str: str,
        *,
        selected: Dict[str, Tuple[int, int, List[str]]],
    ) -> Dict:
        """Merge user metadata (if any) with PowerScale metadata.

        The optional ``file_metadata`` callback (passed at construction) lets
        callers attach arbitrary extra keys to each document — for example a
        project tag or a content-type hint.  PowerScale's own fields (``source``,
        ``snapshot``, ``lin``, ``change_types``) are always written last so they
        cannot be accidentally overwritten by the callback.
        """
        user_meta: Dict = {}
        if self._orig_cb:
            try:
                user_meta = self._orig_cb(path_str) or {}
            except Exception as e:
                _logger.warning("file_metadata failed for %s: %s", path_str, e)

        snapshot, lin, changes = selected[os.path.normpath(path_str)]
        # Start with user-provided keys, then overwrite with PowerScale fields so
        # structural values like `lin` (used for vectorstore deduplication) and
        # `snapshot` (used for checkpointing) are always authoritative.
        meta = dict(user_meta)
        meta.update({"source": path_str, "snapshot": snapshot, "lin": lin, "change_types": changes})
        return meta