import os
import logging
import fnmatch
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.file.base import get_default_fs, _DefaultFileMetadataFunc
from llama_index.core.schema import Document

from .PowerScaleHelper import PowerScaleHelper

_logger = logging.getLogger(__name__)


class PowerScaleSimpleDirectoryReader(SimpleDirectoryReader):
    """LlamaIndex SimpleDirectoryReader using Dell PowerScale MetadataIQ.

    - `input_dir` mode: uses PowerScaleHelper.get_directory_changes().
    - `input_files` mode: uses PowerScaleHelper.get_directory_changes() with helper configured for input_files.
    - Exclusions, required extensions, hidden filtering, and local existence checks are applied
      before delegating to an inner SimpleDirectoryReader to read file contents.

    Note: The PowerScale share must be mounted locally. Both `input_dir` and `input_files`
    paths are checked for local existence at init time, and file contents are read directly
    from the local mount during load. MetadataIQ is used only for change detection.
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
        required_exts: Optional[List[str]] = None,
        num_files_limit: Optional[int] = None,
        exclude_hidden: bool = True,
    ) -> None:
        """Initialize the reader with a PowerScale-backed selection scope.

        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the Elasticsearch index
            es_api_key: api_key for Elasticsearch in hashed (encoded) form
            input_dir: Folder scope to scan; mutually exclusive with `input_files`
            input_files: Explicit list of files to check/load; mutually exclusive with `input_dir`
            exclude: Glob patterns of file paths to skip before reading (e.g. ["*.tmp", "/ifs/data/**/temp.*"])
            recursive: Passed to the base SimpleDirectoryReader (folder mode semantics)
            verify_ssl: Whether to verify SSL certificates for the Elasticsearch client
            force_scan: Force scanning all data regardless of state
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
            file_extractor: Optional mapping of extension → extractor for the inner reader
            file_metadata: Optional callback that returns per-file metadata dicts
            filename_as_id: Use filename as the document id
            encoding: Text encoding hint for the inner reader
            errors: Encoding error handling strategy ("ignore", "strict", etc.).
            required_exts: Only include files whose extension is in this list (e.g., [".pdf"]).
            num_files_limit: Maximum number of files to read this run.
            exclude_hidden: Skip files whose path contains any hidden component. Defaults to True, matching LlamaIndex's SimpleDirectoryReader.
        """
        # Normalize paths before validation and before passing to the base reader so
        # the PowerScale helper and the inner SimpleDirectoryReader agree on the
        # canonical path representation.
        norm_input_dir = os.path.normpath(input_dir) if input_dir is not None else None
        norm_input_files = (
            [os.path.normpath(p) for p in input_files] if input_files is not None else None
        )

        # Validate scope
        if (norm_input_dir is None) == (norm_input_files is None):
            raise ValueError("Select exactly one of `input_dir` or `input_files`.")
        if norm_input_files == []:
            raise ValueError("input_files cannot be empty")
        if norm_input_dir and not norm_input_dir.startswith("/ifs"):
            raise ValueError("input_dir must start with '/ifs'")
        if norm_input_dir and not os.path.isdir(norm_input_dir):
            raise ValueError(f"Directory does not exist: {norm_input_dir}")
        if norm_input_files:
            for p in norm_input_files:
                if not p.startswith("/ifs"):
                    raise ValueError(f"input_files path must start with '/ifs': {p}")
                if not os.path.isfile(p):
                    raise ValueError(f"File does not exist: {p}")

        # Call the base __init__ that comes after SimpleDirectoryReader so we do
        # not trigger its _add_files() walk of the local filesystem. We then set
        # the attributes SimpleDirectoryReader would have set, which the inner
        # reader created by _child_reader relies on.
        super(SimpleDirectoryReader, self).__init__()

        self.errors = errors
        self.encoding = encoding
        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.exclude_empty = False
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit
        self.raise_on_error = False
        self.file_extractor = file_extractor or {}
        self.fs = get_default_fs()
        self.file_metadata = file_metadata or _DefaultFileMetadataFunc(self.fs)
        self.filename_as_id = filename_as_id

        if norm_input_files:
            self.input_dir = None
            self.input_files = [Path(p) for p in norm_input_files]
        else:
            self.input_dir = Path(norm_input_dir) if norm_input_dir else None
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
        """Apply local filters (exclude, existence, extension allow-list, hidden)."""
        p = os.path.normpath(path_str)

        if self._exclude_exact and any(
            self._match_exclude_pattern(p, pattern) for pattern in self._exclude_exact
        ):
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
            required_exts=self.required_exts,
            num_files_limit=None,
            file_extractor=self.file_extractor,
            file_metadata=meta_wrap,
            filename_as_id=self.filename_as_id,
            errors=self.errors,
            encoding=self.encoding,
            exclude_hidden=self.exclude_hidden,
        )

    def _collect_files(self) -> Dict[str, Tuple[int, int, List[str]]]:
        """Use PowerScale to get the filtered file set for this run.

        The generator is consumed until complete so the checkpoint is saved,
        unless num_files_limit is hit and we break early.
        """
        if self._force_scan:
            file_generator = self.__helper.get_directory_changes(snapshot_id=0)
        else:
            file_generator = self.__helper.get_directory_changes()

        selected: Dict[str, Tuple[int, int, List[str]]] = {}

        for file_path, snapshot, lin, changes in file_generator:
            p = os.path.normpath(str(file_path))
            if self._filter(p):
                selected[p] = (int(snapshot), int(lin), list(changes))
                if self.num_files_limit is not None and len(selected) >= self.num_files_limit:
                    # Intentionally break early WITHOUT fully consuming file_generator.
                    # get_directory_changes() only advances the checkpoint when its
                    # generator is exhausted; abandoning it mid-stream leaves the
                    # checkpoint unchanged.  On the next run the same files are
                    # returned again (together with any newly-changed files), which
                    # avoids silently dropping files that were never yielded to the
                    # caller.
                    break

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
        selected = self._collect_files()
        if not selected:
            _logger.debug("No files matched PowerScale selection or local filters.")
            return []

        ordered = sorted(selected.keys())
        child = self._reader_for(ordered, selected)
        # The file_metadata partial passed to the child is not picklable because it
        # captures the PowerScaleSimpleDirectoryReader instance (which holds an
        # Elasticsearch client). num_workers > 1 triggers multiprocessing.spawn and
        # would crash. The public connector does not expose num_workers, so we keep
        # the SimpleDirectoryReader-compatible signature but force sequential loading.
        if isinstance(num_workers, int) and num_workers > 1:
            _logger.warning(
                "num_workers > 1 is not supported by PowerScaleSimpleDirectoryReader; "
                "using sequential load."
            )
            num_workers = None
        return child.load_data(
            show_progress=show_progress, num_workers=num_workers, fs=fs
        )

    def lazy_load_data(
        self, show_progress: bool = False, **kwargs: Any
    ) -> Iterator[Document]:
        """Yield Documents one file at a time.

        The MetadataIQ scan and checkpoint write complete before any Document is
        yielded to the caller. If document loading or downstream processing fails
        mid-stream, the checkpoint will already have advanced and affected files will
        not be reprocessed on the next run. This is a known limitation of the current
        single-phase checkpoint design.

        When num_files_limit is set, the generator is stopped early once the limit is
        reached. The checkpoint is NOT advanced in this case, so the same files will
        be returned again on the next run along with any new files. This is intentional:
        it prevents silent data loss that would occur if the checkpoint advanced past
        files that were never yielded to the caller.
        """
        selected = self._collect_files()
        if not selected:
            _logger.debug("No files matched PowerScale selection or local filters.")
            return

        ordered = sorted(selected.keys())
        child = self._reader_for(ordered, selected)
        # SimpleDirectoryReader v0.14.12 has no lazy_load_data() override, but it does
        # provide iter_data() which yields a list of Documents per file.
        for docs in child.iter_data(show_progress=show_progress):
            yield from docs

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
                _logger.debug("file_metadata failed for %s: %s", path_str, e)

        snapshot, lin, changes = selected[os.path.normpath(path_str)]
        # Start with user-provided keys, then overwrite with PowerScale fields so
        # structural values like `lin` (used for vectorstore deduplication) and
        # `snapshot` (used for checkpointing) are always authoritative.
        meta = dict(user_meta)
        meta.update({"source": path_str, "snapshot": snapshot, "lin": lin, "change_types": changes})
        return meta