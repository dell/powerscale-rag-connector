import copy
import logging
import json

from collections.abc import Mapping
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

    **Checkpointing** — callers can persist the highest-seen snapshot ID by
    calling :meth:`save_checkpoint` (keyed by ``app_name``).  On the next run
    only files whose snapshot ID is strictly greater than the saved value are
    returned, making scans incremental.  Passing ``snapshot_id=0`` to the
    iterators bypasses the checkpoint and returns files from snapshot 1 onwards
    (i.e. ``metadata.snapshots.s2 > 0``); snapshot 0 files are not returned.

    **Key methods:**

    * :meth:`get_directory_changes` — primary iterator; yields
      ``(Path, snapshot, lin, change_types)`` tuples. It does **not** advance the
      checkpoint by default: pass ``save_checkpoint=True`` to write it once the
      generator is exhausted, or call :meth:`save_checkpoint` yourself after
      downstream ingestion has succeeded.
    * :meth:`get_new_files` — thin wrapper that filters to ``ENTRY_ADDED`` only;
      it also leaves the checkpoint untouched.
    * :meth:`get_all_files` — full-scan iterator returning every file regardless
      of change type.
    * :meth:`build_query` — constructs the Elasticsearch query for the active scope.
    * :meth:`save_checkpoint` / :meth:`get_checkpoint` — explicit checkpoint I/O.
    """

    # Attempts for a checkpoint write that loses an optimistic-concurrency race.
    __CHECKPOINT_WRITE_ATTEMPTS = 3

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
        self.__files_seen: bool = False  # True after get_directory_changes yields at least one file
        # Elasticsearch document version of the loaded checkpoint, used for
        # optimistic concurrency control on write.  None until a checkpoint is read.
        self.__checkpoint_seq_no: Optional[int] = None
        self.__checkpoint_primary_term: Optional[int] = None
        self.__app_name = app_name
        self.__app_version = app_version

        # checkpoint document names (written to user's elastic index)
        self.__document_name = self.__app_name
        self.__dataset_index = "powerscale_rag_datasets"

        # Require exactly one scope at construction time
        if (self.__dataset_name is None) and (self.__folder_path is None) and (self.__input_files is None):
            raise ValueError(
                "select one of dataset_name, folder_path, or input_files as iteration scope"
            )
        if self.__folder_path is not None and self.__input_files is not None:
            raise ValueError("folder_path and input_files are mutually exclusive; select one only")
        if self.__dataset_name is not None and (self.__folder_path is not None or self.__input_files is not None):
            raise ValueError("dataset_name is mutually exclusive with folder_path and input_files; select one only")

        # Normalize folder_path once, before validation, so the checkpoint key, the
        # Elasticsearch query and the boundary post-filter all agree on one value.
        if self.__folder_path is not None:
            self.__normalized_folder = self.__folder_path.strip().rstrip("/")
        else:
            self.__normalized_folder = None

        # folder_path must be an absolute PowerScale path beginning with /ifs.
        if self.__normalized_folder is not None and not self.__normalized_folder.startswith("/ifs"):
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

        # Exact-match set used to post-filter results for the input_files scope.
        # build_query() uses match_phrase on the analyzed data.path field, which also
        # matches descendant and adjacent-token paths, so hits must be narrowed here.
        self.__input_files_set = (
            set(self.__input_files) if self.__input_files is not None else None
        )

        self.__es = Elasticsearch(
            hosts=self.__es_host_url,
            api_key=self.__es_api_key,
            verify_certs=self.__verify_ssl,
            ssl_show_warn=not self.__verify_ssl,
        )

        # Load the MetadataIQ dataset definition when configured.
        if self.__dataset_name is not None:
            self.__dataset_doc = self.refresh_dataset()

        # Configure checkpoint root key from the active scope
        if self.__dataset_name is not None:
            self.__checkpoint_root = "datasets"
            self.__checkpoint_key = "dataset"
            self.__checkpoint_value = self.__dataset_name
        elif self.__folder_path is not None:
            self.__checkpoint_root = "folder_paths"
            self.__checkpoint_key = "path"
            self.__checkpoint_value = self.__normalized_folder  # use normalized path
        elif self.__input_files is not None:
            self.__checkpoint_root = "input_files"
            self.__checkpoint_key = "paths"
            self.__checkpoint_value = sorted(self.__input_files)  # normalize order for stable matching
        else:
            raise ValueError("Could not determine checkpoint root configuration")

        # Load or initialize the checkpoint document.
        ckpt_success, self.__last_state = self.get_checkpoint()

        _masked_key = "***"
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
            # Record the document version for optimistic concurrency control on write.
            self.__checkpoint_seq_no = resp.get("_seq_no")
            self.__checkpoint_primary_term = resp.get("_primary_term")
            return True, self.__last_state
        except exceptions.NotFoundError:
            self.__last_state = self.init_checkpoint_doc()
            self.__checkpoint_seq_no = None
            self.__checkpoint_primary_term = None
            return False, self.__last_state

    def _scope_matches(self, keydoc: Dict[str, Any]) -> bool:
        """Return True when a checkpoint entry belongs to this helper's scope.

        For folder scopes, entries written by older versions may carry an
        unnormalized path (e.g. a trailing slash or surrounding whitespace), so
        compare on the normalized form to avoid a spurious full re-ingest.
        """
        if keydoc.get("version") != self.__app_version:
            return False
        stored = keydoc.get(self.__checkpoint_key)
        if self.__normalized_folder is not None and isinstance(stored, str):
            return stored.strip().rstrip("/") == self.__normalized_folder
        return stored == self.__checkpoint_value

    def _in_scope(self, file_path: str) -> bool:
        """Return True when an Elasticsearch hit really belongs to the active scope.

        Both scope queries run against the *analyzed* ``data.path`` field, so
        Elasticsearch returns more than the caller asked for:

        * ``folder_path`` uses ``match_phrase_prefix``, so a query for
          ``/ifs/data/foo`` also matches the sibling ``/ifs/data/foobar``.
        * ``input_files`` uses ``match_phrase``, so a request for
          ``/ifs/data/report`` also matches ``/ifs/data/report/inner.txt`` and
          ``/ifs/data/report.bak``.

        The ``dataset_name`` scope is user-defined and is intentionally not
        narrowed here.
        """
        if self.__normalized_folder is not None:
            if file_path != self.__normalized_folder and not file_path.startswith(
                self.__normalized_folder + "/"
            ):
                _logger.debug(
                    "Skipping file outside folder scope: %s (folder: %s)",
                    file_path,
                    self.__normalized_folder,
                )
                return False
            return True

        if self.__input_files_set is not None:
            if file_path not in self.__input_files_set:
                _logger.debug(
                    "Skipping file not in requested input_files: %s", file_path
                )
                return False
            return True

        return True

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
        self.__dataset_doc = resp["_source"]
        return self.__dataset_doc

    def update_latest_snapid(self) -> None:
        """Update the latest snapshot ID from the index"""
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

    def get_snapshot_id(self) -> int:
        """Look up last processed snapshot id for current path and version"""
        if self.__last_state is None:
            self.get_checkpoint()

        try:
            checkpoints = self.__last_state[self.__checkpoint_root]
        except KeyError:
            return -1

        for ckpt in checkpoints:
            if self._scope_matches(ckpt):
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
            if self._scope_matches(ckpt):
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

    def __build_checkpoint_state(self) -> Tuple[Dict[str, Any], bool]:
        """Return (candidate state, needs_write) for the current scope.

        Works on a copy of the loaded checkpoint so a skipped write never leaves
        the in-memory state diverged from what is persisted in Elasticsearch.
        """
        if self.__last_state is None:
            _logger.debug(
                "Checkpoint save, no current last_state, creating new save document"
            )
            state = self.init_checkpoint_doc()
        else:
            state = copy.deepcopy(self.__last_state)
            # Backfill missing root keys so a checkpoint written by an older version
            # (or a different scope sharing the same app_name) can still be appended to.
            for key, value in self.init_checkpoint_doc().items():
                if key not in state:
                    state[key] = value

        state_key_found = False
        state_index = -1
        state_snapshot_id = -1  # use -1 so a brand-new entry always triggers a write
        old_saved_mtime = None
        # Locate the existing checkpoint entry for this scope and version.
        for index, keydoc in enumerate(state[self.__checkpoint_root]):
            if self._scope_matches(keydoc):
                state_index = index
                state_snapshot_id = keydoc.get("snapshot", -1)
                old_saved_mtime = keydoc.get("saved_mtime")
                state_key_found = True
                break

        # Preserve saved_mtime on empty runs to avoid misclassifying modified files as added.
        # __files_seen is True only after get_directory_changes yields at least one file.
        if self.__files_seen:
            saved_mtime = self.__max_mtime
        elif state_key_found and old_saved_mtime is not None:
            saved_mtime = old_saved_mtime
        else:
            # get_directory_changes ran but saw no files and there is no prior saved_mtime
            saved_mtime = None

        doc = {
            self.__checkpoint_key: self.__checkpoint_value,
            "version": self.__app_version,
            "snapshot": self.__latest_snapshot_id,
        }
        if saved_mtime is not None:
            doc["saved_mtime"] = saved_mtime

        if state_key_found:
            state[self.__checkpoint_root][state_index] = doc
        else:
            # Append a new checkpoint entry for this scope.
            state[self.__checkpoint_root].append(doc)

        needs_write = state_snapshot_id < self.__latest_snapshot_id
        if not needs_write:
            _logger.debug(
                "Skipping checkpoint write for %s %s (version %d), latest_snapshot_id = %d, state_snapshot_id=%d",
                self.__checkpoint_key,
                self.__checkpoint_value,
                self.__app_version,
                self.__latest_snapshot_id,
                state_snapshot_id,
            )
        return state, needs_write

    def save_checkpoint(self) -> None:
        """Create or Update the last run numbers so future calls know where we last indexed"""
        if self.__latest_snapshot_id < 0:
            _logger.warning(
                "Skipping checkpoint write: latest_snapshot_id is invalid (%d), "
                "ES may be unavailable",
                self.__latest_snapshot_id,
            )
            return

        for attempt in range(1, self.__CHECKPOINT_WRITE_ATTEMPTS + 1):
            state, needs_write = self.__build_checkpoint_state()
            if not needs_write:
                # Nothing was persisted, so leave the in-memory state as loaded.
                return

            _logger.debug(
                "Updating checkpoint with %s %s (version %d), snapshot_id = %d",
                self.__checkpoint_key,
                self.__checkpoint_value,
                self.__app_version,
                self.__latest_snapshot_id,
            )
            # Optimistic concurrency control: fail instead of blindly overwriting a
            # checkpoint another writer advanced since we loaded it.
            if self.__checkpoint_seq_no is not None and self.__checkpoint_primary_term is not None:
                occ = {
                    "if_seq_no": self.__checkpoint_seq_no,
                    "if_primary_term": self.__checkpoint_primary_term,
                }
            else:
                # No document was found on load, so require this write to create it.
                # If a concurrent first run created it meanwhile, this conflicts and
                # the retry reloads that document instead of overwriting it.
                occ = {"op_type": "create"}
            try:
                resp = self.__es.index(
                    index=self.__es_index_name,
                    id=self.__document_name,
                    document=state,
                    **occ,
                )
            except exceptions.ConflictError:
                _logger.warning(
                    "Checkpoint write conflicted with a concurrent writer "
                    "(attempt %d/%d); reloading checkpoint and retrying",
                    attempt,
                    self.__CHECKPOINT_WRITE_ATTEMPTS,
                )
                self.get_checkpoint()
                continue
            # Track the new document version so a later save in this process still
            # participates in optimistic concurrency control.
            if isinstance(resp, Mapping):
                self.__checkpoint_seq_no = resp.get("_seq_no")
                self.__checkpoint_primary_term = resp.get("_primary_term")
            self.__last_state = state
            return

        _logger.error(
            "Checkpoint not saved for %s %s: %d concurrent write conflicts. "
            "The next run will rescan from the last successfully saved snapshot.",
            self.__checkpoint_key,
            self.__checkpoint_value,
            self.__CHECKPOINT_WRITE_ATTEMPTS,
        )

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
                                "data.path": "/ifs/<path>"
                            }
                        }
                    ]
                }
            }
        """
        if self.__normalized_folder is not None:
            # match_phrase_prefix on the analyzed data.path field may also match
            # sibling directories (e.g. /ifs/data/foo can match /ifs/data/foobar),
            # so get_directory_changes applies a post-filter to enforce the boundary.
            base_query = {
                "bool": {"must": [{"match_phrase_prefix": {"data.path": self.__normalized_folder}}]}
            }
        elif self.__dataset_name is not None:
            # use dataset definition query; tolerate both string-JSON and dict storage
            raw_query = self.__dataset_doc["query"]
            if isinstance(raw_query, str):
                parsed = json.loads(raw_query)
            else:
                parsed = raw_query
            if isinstance(parsed, dict) and "query" in parsed:
                # Full search request body was stored; use the inner query clause.
                user_query = parsed["query"]
            else:
                user_query = parsed

            # Reuse an existing bool query; otherwise wrap the user query in bool.must
            # before appending the connector's filters.
            if isinstance(user_query, dict) and "bool" in user_query:
                base_query = copy.deepcopy(user_query)
                # bool.must may be a single dict in valid ES DSL; normalize to a list.
                if "must" in base_query["bool"] and not isinstance(base_query["bool"]["must"], list):
                    base_query["bool"]["must"] = [base_query["bool"]["must"]]
                # Elasticsearch defaults minimum_should_match to 1 only while a bool
                # query has no must/filter clauses.  The connector appends must
                # clauses below, which would silently flip that default to 0 and turn
                # a should-only dataset filter into a no-op, so pin it to 1 here.
                bool_clause = base_query["bool"]
                if (
                    bool_clause.get("should")
                    and "minimum_should_match" not in bool_clause
                    and not bool_clause.get("must")
                    and not bool_clause.get("filter")
                ):
                    bool_clause["minimum_should_match"] = 1
            else:
                clauses = user_query if isinstance(user_query, list) else [user_query]
                base_query = {"bool": {"must": clauses}}
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

        # Restrict results to regular files (exclude directories and symlinks).
        base_conditions = [{"term": {"data.file_type": "regular"}}]

        # Add the snapshot range filter when not scanning every file.
        if not all_files:
            base_conditions.append({"range": {"metadata.snapshots.s2": {"gt": snapshot_id}}})
            base_conditions.append(
                {"range": {"metadata.snapshots.s2": {"lte": self.__latest_snapshot_id}}}
            )

        # Append the connector-level conditions to the base bool query.
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

        # Upper-bound the scan at the latest known snapshot so files added during
        # processing are deferred to the next run.
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
            save_checkpoint: when True (default), the checkpoint is written once the
                generator is fully exhausted. Set to False to defer the write and call
                :meth:`save_checkpoint` manually after downstream work completes.

        Returns:
            Iterator yielding tuples containing:
            - Path: pathlib.Path object of the file
            - snapshot: MetadataIQ snapshot number
            - lin: OneFS logical inode number
            - change_types: List of changes (e.g. ['ENTRY_ADDED'], ['ENTRY_MODIFIED'])
        """
        search_success = False
        try:
            # A missing checkpoint means nothing has been ingested yet, so this is a
            # first run regardless of the requested start snapshot. Relying on the
            # checkpoint alone keeps force_scan (snapshot_id=0) on a fresh checkpoint
            # consistent with a normal first run.
            is_first_run = self.get_snapshot_id() < 0
            raw_saved_mtime = self._get_saved_mtime_optional()
            saved_mtime = raw_saved_mtime if raw_saved_mtime is not None else 0
            self.__max_mtime = saved_mtime  # preserve previous value if scan returns no results
            self.__files_seen = False
            for document in self.match_files_by_snapshot(snapshot_id):
                _logger.debug("ES returned the following document: %s", document)
                file_path = document["_source"]["data"]["path"]

                # Narrow analyzed-field over-matches down to the requested scope.
                if not self._in_scope(file_path):
                    continue

                snapshot = int(document["_source"]["metadata"]["snapshots"]["s2"])
                lin = int(document["_source"]["data"]["lin"])
                change_types = document["_source"]["data"].get("change_types") or []
                if "ENTRY_MODIFIED" in change_types and is_first_run:
                    change_types = ["ENTRY_ADDED"]
                # Use ``or 0`` so a JSON null/None in the ES document is treated as 0.
                btime = int(document["_source"]["data"].get("btime") or 0)
                mtime = int(document["_source"]["data"].get("mtime") or 0)
                # A file created after the previous run but modified before this scan may be
                # reported as ENTRY_MODIFIED. Reclassify it as ENTRY_ADDED so callers treat
                # it as a newly discovered file.  Do not reclassify when there is no prior
                # saved_mtime (old v1 checkpoints), to avoid misclassifying every modification.
                if (
                    "ENTRY_MODIFIED" in change_types
                    and raw_saved_mtime is not None
                    and btime > saved_mtime
                ):
                    change_types = ["ENTRY_ADDED"]
                self.__max_mtime = max(self.__max_mtime, mtime)
                self.__files_seen = True
                yield Path(file_path), snapshot, lin, change_types
            search_success = True  # only reached if loop ran to completion
        except Exception as e:
            _logger.error(
                "get_directory_changes() failed: scan incomplete, checkpoint not updated. "
                "Error: %s",
                e,
                exc_info=True,
            )
            raise  # Re-raise to prevent partial scans from appearing complete
        finally:
            if search_success and save_checkpoint:
                self.save_checkpoint()

    def get_new_files(self, snapshot_id: int = -1) -> Iterator[Tuple[Path, int, int]]:
        """Return iterator of only files that were added

        The checkpoint is advanced when the generator is fully exhausted.

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

            # Apply the same scope boundary filter as get_directory_changes.
            if not self._in_scope(file_path):
                continue

            snapshot = int(document["_source"]["metadata"]["snapshots"]["s2"])
            lin = int(document["_source"]["data"]["lin"])
            yield Path(file_path), snapshot, lin
