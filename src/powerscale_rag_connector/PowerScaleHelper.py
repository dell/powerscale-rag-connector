import copy
import logging
import json

from typing import Iterator, Tuple, Dict, Any, Optional, List
from pathlib import Path

from elasticsearch import Elasticsearch, exceptions

_logger = logging.getLogger(__name__)


class PowerScaleHelper:
    """Core helper for querying Dell PowerScale MetadataIQ via Elasticsearch.

    MetadataIQ indexes file-system change events into Elasticsearch.  This class
    reads those events incrementally and exposes them as Python iterators so that
    downstream loaders only process files that have actually changed.

    **Scopes** — exactly one must be supplied at construction time:

    * ``folder_path`` — monitor all files under a PowerScale directory path
      (must start with ``/ifs``).
    * ``dataset_name`` — use a named MetadataIQ dataset definition stored in
      Elasticsearch to filter files.
    * ``input_files`` — monitor a fixed list of specific file paths
      (each must start with ``/ifs``).

    **Checkpointing** — after each successful scan the highest-seen snapshot ID
    is written back to Elasticsearch (keyed by ``app_name``).  On the next run
    only files whose snapshot ID is strictly greater than the saved value are
    returned, making scans incremental.  A ``force_scan`` / ``snapshot_id=0``
    call bypasses the checkpoint and returns files from snapshot 1 onwards (i.e.
    ``metadata.snapshots.s2 > 0``); snapshot 0 files are not returned.

    **Key methods:**

    * :meth:`get_directory_changes` — primary iterator; yields
      ``(Path, snapshot, lin, change_types)`` tuples and saves a checkpoint when
      the generator is exhausted normally.
    * :meth:`get_new_files` — thin wrapper that filters to ``ENTRY_ADDED`` only.
    * :meth:`get_all_files` — full-scan iterator returning every file regardless
      of change type.
    * :meth:`build_query` — constructs the Elasticsearch query for the active scope.
    * :meth:`save_checkpoint` / :meth:`get_checkpoint` — explicit checkpoint I/O.
    """

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """Initialize the helper with a scope (folder path, dataset name, or file list).

        Args:
            es_host_url: URI of the Elasticsearch database incl. port (e.g. http://localhost:9200)
            es_index_name: name of the index
            es_api_key: api_key for Elasticsearch in hashed form
            folder_path: The root of the directory tree to search
            input_files: List of specific file paths to process
            dataset_name: The name of the MetadataIQ dataset to load. Note: dataset_name and folder_path are mutually exclusive
            verify_ssl: Whether to verify SSL certificates for Elasticsearch connection. Defaults to True.
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
        """
        self.__es_host_url = es_host_url
        self.__es_index_name = es_index_name
        self.__es_api_key = es_api_key
        self.__folder_path = folder_path
        self.__input_files = input_files
        self.__dataset_name = dataset_name
        self.__last_state = None
        self.__verify_ssl = verify_ssl
        self.__latest_snapshot_id = -1
        self.__max_mtime = 0
        self.__app_name = app_name
        self.__app_version = app_version

        # checkpoint document names (written to user's elastic index)
        self.__document_name = self.__app_name
        self.__dataset_index = "powerscale_rag_datasets"  # MetadataIQ elastic index for storing dataset definitions

        # Throw a ValueError if no valid scope was provided
        if (self.__dataset_name is None) and (self.__folder_path is None) and (self.__input_files is None):
            raise ValueError(
                "select one of dataset_name, folder_path, or input_files as iteration scope"
            )
        if self.__folder_path is not None and self.__input_files is not None:
            raise ValueError("folder_path and input_files are mutually exclusive; select one only")
        if self.__dataset_name is not None and (self.__folder_path is not None or self.__input_files is not None):
            raise ValueError("dataset_name is mutually exclusive with folder_path and input_files; select one only")

        # Validate folder_path begins with "/ifs". The query uses phrase_prefix on the
        # tokenized path field, so anchoring the path to the OneFS root keeps matches unambiguous.
        if self.__folder_path is not None and not self.__folder_path.startswith("/ifs"):
            raise ValueError("folder_path must start with '/ifs'")

        # Validate the list of input_files if it exists
        if self.__input_files is not None:
            if not isinstance(self.__input_files, list) or not all(isinstance(p, str) for p in self.__input_files):
                raise ValueError("input_files must be a List[str]")
            if not self.__input_files:
                raise ValueError("input_files cannot be empty")
            for p in self.__input_files:
                if not p.startswith("/ifs"):
                    raise ValueError("all input_files must start with '/ifs'")

        self.__es = Elasticsearch(
            hosts=self.__es_host_url,
            api_key=self.__es_api_key,
            verify_certs=self.__verify_ssl,
            ssl_show_warn=not self.__verify_ssl,
        )

        # read dataset definition if we have a dataset name
        if self.__dataset_name is not None:
            self.__dataset_doc = self.refresh_dataset()

        # set root key for checkpoint document, folder_path or dataset based on current config
        if self.__dataset_name is not None:
            self.__checkpoint_root = "datasets"
            self.__checkpoint_key = "dataset"
            self.__checkpoint_value = self.__dataset_name
        elif self.__folder_path is not None:
            self.__checkpoint_root = "folder_paths"
            self.__checkpoint_key = "path"
            self.__checkpoint_value = self.__folder_path
        elif self.__input_files is not None:
            self.__checkpoint_root = "input_files"
            self.__checkpoint_key = "paths"
            self.__checkpoint_value = sorted(self.__input_files)  # normalize order for stable matching
        else:
            raise ValueError("Could not determine checkpoint root configuration")

        # initialize current checkpoint document
        ckpt_success, self.__last_state = self.get_checkpoint()

        _masked_key = ("***" + self.__es_api_key[-4:]) if self.__es_api_key else "***"
        _logger.debug(
            "Hostname=%s Index=%s es_api_key=%s folder_path=%s dataset_name=%s last_state=%s",
            self.__es_host_url,
            self.__es_index_name,
            _masked_key,
            self.__folder_path,
            self.__dataset_name,
            self.__last_state,
        )


    def get_checkpoint(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Determine if a past run was performed.
        If a past run was performed, get the previously saved document and return it.
        If no past run was performed, returns an empty state; a full scan will be triggered.
        """
        _logger.debug("Checking Elasticsearch for past runs")
        try:
            resp = self.__es.get(index=self.__es_index_name, id=self.__document_name)
            _logger.debug("Checkpoint query response: %s", resp)
            self.__last_state = resp["_source"]
            return True, self.__last_state
        except exceptions.NotFoundError:
            self.__last_state = self.init_checkpoint_doc()
            return False, self.__last_state

    def refresh_dataset(self) -> Dict[str, Any]:
        """Refresh the MetadataIQ dataset configuration from elastic.
        Returns an empty dict if this helper was not constructed with a dataset_name.
        Raises NotFoundError if the dataset record does not exist in Elasticsearch.
        """
        _logger.debug("Checking elastic for dataset %s", self.__dataset_name)
        if self.__dataset_name is None:
            return {}
        resp = self.__es.get(index=self.__dataset_index, id=self.__dataset_name)
        _logger.debug("Dataset query response: %s", resp)
        return resp["_source"]

    def update_latest_snapid(self) -> None:
        """Update the latest snapshot ID from the index"""
        try:
            result = self.__es.search(
                index=self.__es_index_name,
                aggs={"max_snapid": {"max": {"field": "metadata.snapshots.s2"}}},
                size=0,
            )
            raw = result["aggregations"]["max_snapid"]["value"]
            if raw is None:
                _logger.warning("update_latest_snapid: index exists but has no documents; defaulting to -1")
                self.__latest_snapshot_id = -1
            else:
                self.__latest_snapshot_id = int(raw)
            _logger.debug(
                "MetadataIQ latest snapshot id for %s = %d",
                self.__es_index_name, self.__latest_snapshot_id
            )
        except Exception as e:
            _logger.error(
                "Error in PowerScale RAG Connector Helper update_latest_snapid; defaulting to -1: %s",
                e
            )
            self.__latest_snapshot_id = -1

    def get_snapshot_id(self) -> int:
        """Look up last processed snapshot id for current path and version"""
        if self.__last_state is None:
            self.get_checkpoint()

        try:
            checkpoints = self.__last_state[self.__checkpoint_root]
        except KeyError:
            return -1

        for ckpt in checkpoints:
            # Match both path and version
            if (
                ckpt[self.__checkpoint_key] == self.__checkpoint_value
                and ckpt.get("version") == self.__app_version
            ):
                return ckpt.get("snapshot", -1)

        # not found, return -1
        return -1

    def _get_saved_mtime_optional(self) -> Optional[int]:
        """Look up last processed max mtime, returning None when the field is absent.

        Distinguishes an old v1 checkpoint that did not store saved_mtime from a
        genuine saved_mtime value of 0.
        """
        if self.__last_state is None:
            self.get_checkpoint()

        try:
            checkpoints = self.__last_state[self.__checkpoint_root]
        except KeyError:
            return None

        for ckpt in checkpoints:
            if (
                ckpt[self.__checkpoint_key] == self.__checkpoint_value
                and ckpt.get("version") == self.__app_version
            ):
                return ckpt.get("saved_mtime")

        return None

    def get_saved_mtime(self) -> int:
        """Look up last processed max mtime for current path and version"""
        return self._get_saved_mtime_optional() or 0

    def init_checkpoint_doc(self) -> Dict[str, Any]:
        """Return a new checkpoint document with the known root keys.

        Note: the keys here must match the possible __checkpoint_root values.
        """
        return {"folder_paths": [], "datasets": [], "input_files": []}

    def save_checkpoint(self) -> None:
        """Create or Update the last run numbers so future calls know where we last indexed"""
        if self.__latest_snapshot_id < 0:
            _logger.warning(
                "Skipping checkpoint write: latest_snapshot_id is invalid (%d), "
                "ES may be unavailable",
                self.__latest_snapshot_id,
            )
            return
        if self.__last_state is None:
            _logger.debug(
                "Checkpoint save, no current last_state, creating new save document"
            )
            self.__last_state = self.init_checkpoint_doc()
        else:
            # Backfill missing root keys so a checkpoint written by an older version
            # (or a different scope sharing the same app_name) can still be appended to.
            defaults = self.init_checkpoint_doc()
            for key in defaults:
                if key not in self.__last_state:
                    self.__last_state[key] = defaults[key]

        doc = {
            self.__checkpoint_key: self.__checkpoint_value,
            "version": self.__app_version,
            "snapshot": self.__latest_snapshot_id,
            "saved_mtime": self.__max_mtime,
        }
        state = self.__last_state
        state_key_found = False
        state_snapshot_id = -1  # use -1 so a brand-new entry always triggers a write
        # Check if state has our old run, if we do, update it
        for index, keydoc in enumerate(state[self.__checkpoint_root]):
            # Match both path and version
            if (
                keydoc[self.__checkpoint_key] == self.__checkpoint_value
                and keydoc.get("version") == self.__app_version
            ):
                state_snapshot_id = keydoc.get("snapshot", -1)
                state[self.__checkpoint_root][index] = doc
                state_key_found = True
                break

        # Brand new run, need to add it to our state
        if not state_key_found:
            state[self.__checkpoint_root].append(doc)

        if state_snapshot_id >= self.__latest_snapshot_id:
            _logger.debug(
                "Skipping checkpoint write for %s %s (version %d), latest_snapshot_id = %d, state_snapshot_id=%d",
                self.__checkpoint_key,
                self.__checkpoint_value,
                self.__app_version,
                self.__latest_snapshot_id,
                state_snapshot_id,
            )
        else:
            _logger.debug(
                "Updating checkpoint with %s %s (version %d), snapshot_id = %d",
                self.__checkpoint_key,
                self.__checkpoint_value,
                self.__app_version,
                self.__latest_snapshot_id,
            )
            self.__es.index(
                index=self.__es_index_name,
                id=self.__document_name,
                document=state,
            )
        self.__last_state = state

    def es_search_paged(self, query, batch_size=10000) -> Iterator[Dict[str, Any]]:
        """
        Generates MetadataIQ entries in elasticsearch one at a time using search_after pagination.

        Args:
            query: The Elasticsearch query parameters.
            batch_size: The number of documents to retrieve in each batch.

        Yields:
            Each document from the Elasticsearch results.
        """

        search_after = None
        while True:
            _logger.debug("es_search query = %s", query)
            response = self.__es.search(
                index=self.__es_index_name,
                size=batch_size,
                query=query,
                source=[
                    "data.path",
                    "data.change_types",
                    "data.btime",
                    "data.mtime",
                    "data.lin",
                    "metadata.snapshots.s2",
                ],
                sort=[
                    {"data.lin": "asc"}
                ],  # consistent sorting on OneFS logical inode number
                search_after=search_after,
                request_timeout=3600,  # allow an hour for the first response on a large data set
            )

            if not response["hits"]["hits"]:
                break  # No more results

            for hit in response["hits"]["hits"]:
                yield hit

            search_after = response["hits"]["hits"][-1]["sort"]

    def build_query(self, all_files=True, snapshot_id=-1) -> Dict[str, Any]:
        """Construct an ES query based on the current helper config (dataset, path, all_files)
        Currently the dataset query strings are stored in the dataset definition;
        Path configurations use a standard base query string which looks like:
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "data.file_type": "regular"
                            }
                        },
                        {
                            "match_phrase_prefix": {
                                "data.path": "/ifs/<path>/"
                            }
                        }
                    ]
                }
            }
        """
        if self.__folder_path is not None:
            # build path query, trimming trailing whitespace and slashes, then
            # re-adding a single trailing slash so /ifs/data/ does not false-match
            # a sibling path like /ifs/databank when the field is a single token.
            path = self.__folder_path.rstrip().rstrip("/") + "/"
            base_query = {
                "bool": {"must": [{"match_phrase_prefix": {"data.path": path}}]}
            }
        elif self.__dataset_name is not None:
            # use dataset definition query; tolerate both string-JSON and dict storage
            raw_query = self.__dataset_doc["query"]
            if isinstance(raw_query, str):
                parsed = json.loads(raw_query)
            else:
                parsed = raw_query
            if isinstance(parsed, dict) and "query" in parsed:
                base_query = parsed["query"]
            else:
                base_query = parsed
            _logger.debug("Dataset query from definition: %s", base_query)
        elif self.__input_files is not None:
            # use should+match_phrase for exact path matches on text field
            shoulds = [{"match_phrase": {"data.path": p}} for p in self.__input_files]
            base_query = {
                "bool": {
                    "must": [],
                    "should": shoulds,
                    "minimum_should_match": 1
                }
            }
        else:
            raise ValueError("build_query: no scope configured (folder_path, dataset_name, or input_files required)")

        # restrict results to 'normal' files, no dirs, links, etc.
        base_conditions = [{"term": {"data.file_type": "regular"}}]

        # if we are filtering by snapshot_id, extend the query filters with the
        # range filter
        if not all_files:
            base_conditions.append({"range": {"metadata.snapshots.s2": {"gt": snapshot_id}}})
            base_conditions.append(
                {"range": {"metadata.snapshots.s2": {"lte": self.__latest_snapshot_id}}}
            )

        # add the extra conditions to the base query as a "must" clause
        retval = copy.deepcopy(base_query)
        for condition in base_conditions:
            retval.setdefault("bool", {}).setdefault("must", []).append(condition)

        return retval

    def match_files_by_snapshot(
        self, snapshot_id: int = -1
    ) -> Iterator[Dict[str, Any]]:
        """
        Return all files whose MetadataIQ snapshot is in the selected range.
        If the snapshot_id argument is negative, files since the most recently saved checkpoint will be returned,
        but the checkpoint will not be updated. Calling this function with the default -1 multiple times will return
        the same files (and potentially new ones) repeatedly.
        """
        if snapshot_id < 0:
            snapshot_id = self.get_snapshot_id()

        # set the upper bound on the query to the highest current snapshot
        # therefore if another snapshot comes along while we are processing
        # we will not return those files until the next run, avoiding
        # skipping or duplicating results
        self.update_latest_snapid()

        # Main repo style: always use the snapshot range, even for snapshot_id=0.
        # snapshot_id=0 means "start from the beginning" (gt 0), not "return all".
        query = self.build_query(all_files=False, snapshot_id=snapshot_id)

        _logger.debug("ES query: %s", query)

        return self.es_search_paged(query=query)

    def get_directory_changes(
        self, snapshot_id: int = -1, save_checkpoint: bool = True
    ) -> Iterator[Tuple[Path, int, int, List[str]]]:
        """Return iterator of tuples of (Path, snapshot, lin, change_types) for files in the current path

        Args:
            snapshot_id: snapshot to start from; negative means use the saved checkpoint.
            save_checkpoint: when True (default) the checkpoint is written once the
                generator is exhausted. Callers that need to perform additional work
                (such as parsing documents) before the checkpoint should be committed
                can set this to False and call :meth:`save_checkpoint` themselves.

        Returns:
            Iterator yielding tuples containing:
            - Path: pathlib.Path object of the file
            - snapshot: MetadataIQ snapshot number
            - lin: OneFS logical inode number
            - change_types: List of changes (e.g. ['ENTRY_ADDED'], ['ENTRY_MODIFIED'])
        """
        search_success = False
        try:
            # Main repo style: first run only when the caller explicitly uses the
            # default negative snapshot_id AND no checkpoint exists yet.
            is_first_run = snapshot_id < 0 and self.get_snapshot_id() < 0
            raw_saved_mtime = self._get_saved_mtime_optional()
            saved_mtime = raw_saved_mtime if raw_saved_mtime is not None else 0
            self.__max_mtime = saved_mtime  # preserve previous value if scan returns no results
            for document in self.match_files_by_snapshot(snapshot_id):
                _logger.debug("ES returned the following document: %s", document)
                file_path = document["_source"]["data"]["path"]
                snapshot = int(document["_source"]["metadata"]["snapshots"]["s2"])
                lin = int(document["_source"]["data"]["lin"])
                change_types = document["_source"]["data"].get("change_types", [])
                if "ENTRY_MODIFIED" in change_types and is_first_run:
                    change_types = ["ENTRY_ADDED"]
                # Use ``or 0`` so a JSON null/None in the ES document is treated as 0.
                btime = int(document["_source"]["data"].get("btime") or 0)
                mtime = int(document["_source"]["data"].get("mtime") or 0)
                # If the file's birth time (creation) is newer than the last-run mtime
                # checkpoint, the file must have been created after the previous run and
                # MetadataIQ tagged it ENTRY_MODIFIED because it was also written before we
                # scanned. Reclassify as ENTRY_ADDED so callers treat it as a new file.
                # A missing saved_mtime (None) indicates an old v1 checkpoint; do not
                # reclassify in that case to avoid treating every modification as an add.
                if (
                    "ENTRY_MODIFIED" in change_types
                    and raw_saved_mtime is not None
                    and btime > saved_mtime
                ):
                    change_types = ["ENTRY_ADDED"]
                self.__max_mtime = max(self.__max_mtime, mtime)
                yield Path(file_path), snapshot, lin, change_types
            search_success = True  # only reached if loop ran to completion
        except Exception as e:
            _logger.error(
                "get_directory_changes() failed: scan incomplete, checkpoint not updated. "
                "Error: %s",
                e,
                exc_info=True,
            )
        finally:
            if search_success and save_checkpoint:
                self.save_checkpoint()

    def get_new_files(self, snapshot_id: int = -1) -> Iterator[Tuple[Path, int, int]]:
        """Return iterator of only files that were added

        Args:
            snapshot_id: snapshot ID to start from. If negative, uses last checkpoint.

        Returns:
            Iterator of (Path, snapshot, lin) tuples for added files
        """
        for path, snapshot, lin, change_types in self.get_directory_changes(snapshot_id):
            if "ENTRY_ADDED" in change_types:
                yield path, snapshot, lin

    def get_deleted_files(self, snapshot_id: int = -1) -> Iterator[Tuple[Path, int, int]]:
        """Not supported: MetadataIQ does not emit ENTRY_DELETED events."""
        raise NotImplementedError(
            "get_deleted_files() is not supported: MetadataIQ does not return ENTRY_DELETED events."
        )

    def get_all_files(self) -> Iterator[Tuple[Path, int, int]]:
        """Return iterator of all files matching the configured path/dataset scope.

        Triggers a scan with snapshot_id=0, ignoring any saved checkpoint. The
        query uses ``metadata.snapshots.s2 > 0`` (MetadataIQ snapshot 0 is not
        included), but all remaining files are returned regardless of change type
        (ENTRY_ADDED, ENTRY_MODIFIED, etc.), unlike get_new_files() which only
        yields files that were added.

        Returns:
            Iterator of (Path, snapshot, lin) tuples for files currently
            visible in MetadataIQ for this scope with snapshot > 0.
        """
        # Use match_files_by_snapshot directly so no checkpoint is written.
        for document in self.match_files_by_snapshot(snapshot_id=0):
            file_path = document["_source"]["data"]["path"]
            snapshot = int(document["_source"]["metadata"]["snapshots"]["s2"])
            lin = int(document["_source"]["data"]["lin"])
            yield Path(file_path), snapshot, lin
