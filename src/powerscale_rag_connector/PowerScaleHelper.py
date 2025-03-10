import logging
import json

from typing import Iterator, Iterable, Tuple, Dict, Any, Optional, List
from pathlib import Path

from elasticsearch import Elasticsearch, exceptions


class PowerScaleHelper:
    """Helper class to talk to PowerScale's MetadataIQ indexed metadata facility and provide fast metadata search results"""

    def __init__(
        self,
        es_host_url: str,
        es_index_name: str,
        es_api_key: str,
        folder_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        verify_ssl: bool = True,
        app_name: str = "powerscale_rag_connector",
        app_version: int = 1,
    ) -> None:
        """Initialize the loader with a file path.

        Args:
            es_host_url: fqdn or IP address of the ElasticSearch database
            es_index_name: name of the index
            es_api_key: api_key for ElasticSearch in hashed form
            folder_path: The root of the directory tree to search
            dataset: The name of the MetadataIQ dataset to load. Note: dataset and folder_path are mutually exclusive
            verify_ssl: Whether to verify SSL certificates for Elasticsearch connection. Defaults to True.
            app_name: A unique application name to use for the checkpoint document. Defaults to "powerscale_rag_connector".
            app_version: A version number for the checkpoint document. Defaults to 1.
        """
        self.__es_host_url = es_host_url
        self.__es_index_name = es_index_name
        self.__es_api_key = es_api_key
        self.__folder_path = folder_path
        self.__dataset_name = dataset_name
        self.__last_state = None
        self.__verify_ssl = verify_ssl
        self.__latest_snapshot_id = -1
        self.__app_name = app_name
        self.__app_version = app_version

        self.__es = Elasticsearch(
            hosts=self.__es_host_url,
            api_key=self.__es_api_key,
            verify_certs=self.__verify_ssl,
            ssl_show_warn=not self.__verify_ssl,
        )

        # checkpoint document names (written to user's elastic index)
        self.__document_name = self.__app_name
        self.__dataset_index = "powerscale_rag_datasets"  # MetadataIQ elastic index for storing dataset definitions

        # Throw a ValueError if caller has requested both a dataset and a folder_path (or neither)
        if (self.__dataset_name is not None) and (self.__folder_path is not None):
            raise ValueError(
                "dataset and folder_path attributes are mutually exclusive; select one only"
            )
        if (self.__dataset_name is None) and (self.__folder_path is None):
            raise ValueError(
                "select one of dataset_name or folder_path as iteration scope"
            )

        # Validate folder_path begins with "/ifs"
        #
        # The MetadataIQ document schema  for OneFS 9.10 uses a path tokenizer for the path/filename string.
        # This complicates query construction somewhat, as it matches search terms against each component of the
        # full path string.
        # In order to deal with this we are searching with elastic's "phrase_prefix" predicate,
        # which will match partial, intermediate paths. This can cause confusing results, so
        # for the sake of this PowerScale demo we are requiring that paths begin with the
        # OneFS convention of a root directory called "/ifs".  MetadataIQ consumers
        # are certainly free to come up with better queries that work with their applications,
        # which can be supported through the dataset query interface.
        if self.__folder_path is not None and not self.__folder_path.startswith("/ifs"):
            raise ValueError("folder_path must start with '/ifs'")

        # read dataset definition if we have a dataset name
        if self.__dataset_name is not None:
            self.__dataset_doc = self.refresh_dataset()
            if self.__dataset_doc is None:
                raise ValueError("Dataset %s not found" % (self.__dataset_name))

        # set root key for checkpoint document, folder_path or dataset based on current config
        if self.__dataset_name is not None:
            self.__checkpoint_root = "datasets"
            self.__checkpoint_key = "dataset"
            self.__checkpoint_value = self.__dataset_name
        elif self.__folder_path is not None:
            self.__checkpoint_root = "folder_paths"
            self.__checkpoint_key = "path"
            self.__checkpoint_value = self.__folder_path
        else:
            raise ValueError("Could not determine checkpoint root configuration")

        # initialize current checkpoint document
        ckpt_success, self.__last_state = self.get_checkpoint()

        logging.debug(
            "Hostname=%s Index=%s es_api_key=%s folder_path=%s dataset_name = %s last_state=%s"
            % (
                self.__es_host_url,
                self.__es_index_name,
                self.__es_api_key,
                self.__folder_path,
                self.__dataset_name,
                self.__last_state,
            )
        )

    def get_checkpoint(self) -> Tuple[bool, Dict[str, Any] | None]:
        """Determine if a past run was performed.
        If a past run was performed, get the previously saved document and return it
        If no past run was performed, snapshot is zero and full index scan would be performed
        """
        logging.debug("Checking ElasticSearch for past runs")
        try:
            resp = self.__es.get(index=self.__es_index_name, id=self.__document_name)
            logging.debug("Checkpoint query response: %s" % (resp))
            self.__last_state = resp["_source"]
            return True, self.__last_state
        except exceptions.NotFoundError as e:
            return False, {self.__checkpoint_root: []}

    def refresh_dataset(self) -> Dict[str, Any]:
        """Refresh the MetadataIQ dataset configuration from elastic
        Throws NotFoundError if dataset record not found
        """
        logging.debug("Checking elastic for dataset %s", self.__dataset_name)
        if self.__dataset_name is None:
            return {}

        resp = self.__es.get(index=self.__dataset_index, id=self.__dataset_name)
        logging.debug("Dataset query response: %s", (resp))
        return resp["_source"]

    def update_latest_snapid(self) -> None:
        """Update the latest snapshot ID from the index"""
        query = {
            "aggs": {"max_snapid": {"max": {"field": "metadata.snapshots.s2"}}},
            "size": 0,
        }

        try:
            result = self.__es.search(index=self.__es_index_name, body=query)
            self.__latest_snapshot_id = int(
                result["aggregations"]["max_snapid"]["value"]
            )
            logging.debug(
                "MetadataIQ latest snapshot id for %s = %d",
                self.__es_index_name, self.__latest_snapshot_id
            )
        except Exception as e:
            logging.error(
                "Error in PowerScale RAG ConnectorHelper.update_latest_snapid; defaulting to gen -1: %s",
                e
            )
            self.__latest_snapshot_id = (
                -1
            )  # set to default value in case of exception and return

    def get_snapshot_id(self) -> int:
        """Look up last processed snapshot id for current path and version"""
        if self.__last_state is None:
            self.get_checkpoint()

        if self.__last_state is None:
            return -1
        else:
            try:
                checkpoints = self.__last_state[self.__checkpoint_root]
            except KeyError:
                self.__last_state = (
                    None  # checkpoint root not found, force refresh of last state
                )
                return -1

        for ckpt in checkpoints:
            # Match both path and version
            if (
                ckpt[self.__checkpoint_key] == self.__checkpoint_value
                and ckpt.get("version") == self.__app_version
            ):
                return ckpt["snapshot"]

        # not found, return -1
        return -1

    def init_checkpoint_doc(self):
        """Initialize new checkpoint document with current set of known checkpoint roots.
        N.B. This routine hardcodes key names and must stay in sync with possible
        __checkpoint_root and __checkpoint_key values
        """
        retval = {"folder_paths": [], "datasets": []}
        retval["folder_paths"].append(
            {"path": "__empty_path__", "version": 1, "snapshot": -1}
        )
        retval["datasets"].append(
            {"dataset": "__empty_dataset__", "version": 1, "snapshot": -1}
        )
        return retval

    def save_checkpoint(self) -> None:
        """Create or Update the last run numbers for so future calls know where we last indexed"""
        # if we have never read or updated a checkpoint, create a new document
        if self.__last_state is None:
            logging.debug(
                "Checkpoint save, no current last_state, creating new save document"
            )
            self.__last_state = self.init_checkpoint_doc()

        doc = {
            self.__checkpoint_key: self.__checkpoint_value,
            "version": self.__app_version,
            "snapshot": self.__latest_snapshot_id,
        }
        state = self.__last_state
        state_key_found = False
        state_snapshot_id = 0
        # Check if state has our old run, if we do, update it
        for index, keydoc in enumerate(state[self.__checkpoint_root]):
            # Match both path and version
            if (
                keydoc[self.__checkpoint_key] == self.__checkpoint_value
                and keydoc.get("version") == self.__app_version
            ):
                state_snapshot_id = keydoc["snapshot"]
                state[self.__checkpoint_root][index] = doc
                state_key_found = True

        # Brand new run, need to add it to our state
        if state_key_found == False:
            state[self.__checkpoint_root].append(doc)

        if state_snapshot_id >= self.__latest_snapshot_id:
            logging.debug(
                "Skipping checkpoint write for %s %s (version %d), latest_snapshot_id = %d, state_snapshot_id=%d"
                % (
                    self.__checkpoint_key,
                    self.__checkpoint_value,
                    self.__app_version,
                    self.__latest_snapshot_id,
                    state_snapshot_id,
                )
            )
        else:
            logging.debug(
                "Updating checkpoint with %s %s (version %d), snapshot_id = %d"
                % (
                    self.__checkpoint_key,
                    self.__checkpoint_value,
                    self.__app_version,
                    self.__latest_snapshot_id,
                )
            )
            self.__es.index(
                index=self.__es_index_name,
                id=self.__document_name,
                document=state,
            )
        self.__last_state = state

    def es_search_paged(self, query, batch_size=10000):
        """
        Generates MetadataIQ entries in elasticsearch one at a time using search_after pagination.

        Args:
            index: Name of the Elasticsearch index (index must contain MetadataIQ data).
            query: The Elasticsearch query parameters.
            batch_size: The number of documents to retrieve in each batch.

        Yields:
            Each document from the Elasticsearch results.
        """

        search_after = None
        while True:
            logging.debug("es_search query = %s" % (query))
            response = self.__es.search(
                index=self.__es_index_name,
                size=batch_size,
                query=query,
                source=[
                    "data.path",
                    "data.change_types",
                    "sort",
                    "data.lin",
                    "metadata.snapshots.s2",
                ],
                sort=[
                    {"data.lin": "asc"}
                ],  # consistent sorting on OneFS logical inode number
                search_after=search_after,
                request_timeout=3600,  # allow an hour for the first response on a large data set
            )

            # logging.debug(response)
            # with open("/tmp/resp.json", "w") as f:
            #     f.write(json.dumps(str(response), indent=4))

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
        retval = {}
        if self.__folder_path is not None:
            # build path query, trimming trailing whitespace and slashes
            path = self.__folder_path.rstrip("/").rstrip()
            base_query = {
                "bool": {"must": [{"match_phrase_prefix": {"data.path": path}}]}
            }
        elif self.__dataset_name is not None:
            # use dataset definition query
            base_query = json.loads(self.__dataset_doc["query"])["query"]
            logging.debug("Dataset query string from definition: %s" % (base_query))

        # restrict results to 'normal' files, no dirs, links, etc.
        base_conditions = [{"term": {"data.file_type": "regular"}}]

        # if we are filtering by snapshot_id, extend the query filters with the
        # range filter
        if all_files == False:
            base_conditions.append({"range": {"metadata.snapshots.s2": {"gt": snapshot_id}}})  # type: ignore
            base_conditions.append(
                {"range": {"metadata.snapshots.s2": {"lte": self.__latest_snapshot_id}}}  # type: ignore
            )

        # add the extra conditions to the base query as a "must" clause
        retval = base_query
        for condition in base_conditions:
            try:
                retval["bool"]["must"].append(condition)  # type: ignore
            except KeyError as e:
                # "must" clause does not exist in base_query, create one
                retval["bool"]["must"] = []
                retval["bool"]["must"].append(condition)  # type: ignore

        return retval

    def match_files_by_snapshot(
        self, snapshot_id: int = -1
    ) -> Iterable[Dict[str, Any]]:
        """
        Return all files that have been added to the current path since the selected snapshot id up to the current snapshot
        If the snapshot_id argument is negative, files since the most recently saved checkpoint will be returned,
        but the checkpoint will not be updated.  Calling this funtion with the default -1 multiple times will return
        the same files (and potentially new ones) repeatedly.
        """
        if snapshot_id < 0:
            snapshot_id = self.get_snapshot_id()

        # set the upper bound on the query to the highest current snapshot
        # therefore if another snapshot comes along while we are processing
        # we will not return those files until the next run, avoiding
        # skipping or duplicating results
        self.update_latest_snapid()

        query = self.build_query(all_files=False, snapshot_id=snapshot_id)

        logging.debug("ES query: %s" % (query))

        return self.es_search_paged(query=query)

    def get_directory_changes(
        self, snapshot_id: int = -1
    ) -> Iterator[Tuple[Path, int, List[str]]]:
        """Return iterator of tuples of (Path, snapshot, change_types) for files in the current path

        Returns:
            Iterator yielding tuples containing:
            - Path: pathlib.Path object of the file
            - snapshot: MetadataIQ snapshot number
            - change_types: List of changes (e.g. ['ENTRY_ADDED'], ['ENTRY_DELETED'])
        """
        try:
            search_success = True
            for document in self.match_files_by_snapshot(snapshot_id):
                logging.debug("ES return the following document: %s" % (document))
                file_path = document["_source"]["data"]["path"]
                snapshot = document["_source"]["metadata"]["snapshots"]["s2"]
                change_types = document["_source"]["data"].get("change_types", [])
                yield Path(file_path), snapshot, change_types
        except Exception as e:
            search_success = False
            logging.error(
                "get_directory_changes() exception; iteration failed, "
                "skipping checkpoint update: %s",
                str(e),
            )
        finally:
            if search_success:
                self.save_checkpoint()

    def get_new_files(self, snapshot_id: int = -1) -> Iterator[Tuple[Path, int]]:
        """Return iterator of only files that were added

        Args:
            snapshot_id: snapshot ID to start from. If negative, uses last checkpoint.

        Returns:
            Iterator of (Path, snapshot) tuples for added files
        """
        for path, snapshot, change_types in self.get_directory_changes(snapshot_id):
            if "ENTRY_ADDED" in change_types:
                yield path, snapshot

    def get_deleted_files(self, snapshot_id: int = -1) -> Iterator[Tuple[Path, int]]:
        """Return iterator of only files that were deleted

        Args:
            snapshot_id: snapshot ID to start from. If negative, uses last checkpoint.

        Returns:
            Iterator of (Path, snapshot) tuples for deleted files
        """
        for path, snapshot, change_types in self.get_directory_changes(snapshot_id):
            if "ENTRY_DELETED" in change_types:
                yield path, snapshot

    def get_all_files(self) -> Iterator[Tuple[Path, int]]:
        """Return iterator of all files matching path/dataset

        Returns:
            Iterator of (Path, snapshot) tuples for added files
        """
        return self.get_new_files(snapshot_id=0)
