"""Tests for PowerScaleHelper with the dataset_name scope.

This covers the code paths that are not exercised by the folder_path / input_files
tests: dataset definition lookup, checkpoint root ('datasets'), refresh_dataset(),
and the restored get_all_files() convenience method.
"""

import json
from pathlib import Path

import pytest

from tests.conftest import FakeElasticsearch, make_hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATASET_QUERY_SIMPLE = json.dumps(
    {"query": {"bool": {"must": [{"term": {"data.tag": "test"}}]}}}
)


def _dataset_fake(**kwargs):
    """Return a FakeElasticsearch pre-loaded with a minimal dataset definition."""
    return FakeElasticsearch(
        dataset_doc={"query": _DATASET_QUERY_SIMPLE},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Init / dataset lookup
# ---------------------------------------------------------------------------

def test_dataset_init_fetches_dataset_definition(make_helper):
    """Helper with dataset_name calls get() on the dataset index at construction."""
    fake = _dataset_fake()
    make_helper(fake, dataset_name="my_dataset")
    dataset_gets = [
        (idx, id_)
        for idx, id_ in fake.get_calls
        if idx == FakeElasticsearch.DATASET_INDEX
    ]
    assert len(dataset_gets) == 1
    assert dataset_gets[0][1] == "my_dataset"


def test_dataset_init_not_found_raises(make_helper):
    """If the dataset definition is absent in Elasticsearch, NotFoundError propagates."""
    from elasticsearch import exceptions
    fake = FakeElasticsearch(dataset_doc=None)  # will raise NotFoundError on dataset get
    with pytest.raises(exceptions.NotFoundError):
        make_helper(fake, dataset_name="missing")


def test_dataset_init_checkpoint_root_is_datasets(make_helper):
    """get_checkpoint() for a dataset-scoped helper reads from the 'datasets' root."""
    fake = _dataset_fake()
    helper = make_helper(fake, dataset_name="ds1")
    _, state = helper.get_checkpoint()
    assert "datasets" in state


# ---------------------------------------------------------------------------
# build_query with dataset scope
# ---------------------------------------------------------------------------

def test_dataset_build_query_includes_definition_terms(make_helper):
    """build_query() merges the dataset definition query with the file_type filter."""
    fake = _dataset_fake()
    helper = make_helper(fake, dataset_name="ds1")
    q = helper.build_query(all_files=True)
    musts = q["bool"]["must"]
    assert {"term": {"data.tag": "test"}} in musts
    assert {"term": {"data.file_type": "regular"}} in musts


def test_dataset_build_query_incremental_adds_range(make_helper):
    """Incremental dataset query includes snapshot range filters."""
    fake = _dataset_fake()
    helper = make_helper(fake, dataset_name="ds1")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    q = helper.build_query(all_files=False, snapshot_id=10)
    musts = q["bool"]["must"]
    assert {"range": {"metadata.snapshots.s2": {"gt": 10}}} in musts
    assert {"range": {"metadata.snapshots.s2": {"lte": 20}}} in musts


# ---------------------------------------------------------------------------
# get_directory_changes / checkpoint save with dataset scope
# ---------------------------------------------------------------------------

def test_dataset_directory_changes_yields_files(make_helper):
    """get_directory_changes() works end-to-end with dataset scope."""
    pages = [[make_hit("/ifs/data/a.txt", lin=1, snapshot=5)]]
    fake = _dataset_fake(search_pages=pages, max_snapid=5)
    helper = make_helper(fake, dataset_name="ds1")
    results = list(helper.get_directory_changes())
    assert results == [(Path("/ifs/data/a.txt"), 5, 1, [])]


def test_dataset_checkpoint_saved_under_datasets_key(make_helper):
    """Checkpoint written after a dataset-scoped run uses the 'datasets' root."""
    pages = [[make_hit("/ifs/data/a.txt", lin=1, snapshot=7)]]
    fake = _dataset_fake(search_pages=pages, max_snapid=7)
    helper = make_helper(fake, dataset_name="ds_alpha")
    list(helper.get_directory_changes(save_checkpoint=True))
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]
    assert "datasets" in written
    entry = written["datasets"][0]
    assert entry["dataset"] == "ds_alpha"
    assert entry["snapshot"] == 7


def test_dataset_get_snapshot_id_reads_datasets_root(make_helper):
    """get_snapshot_id() finds the matching entry under 'datasets'."""
    checkpoint = {
        "datasets": [{"dataset": "ds_beta", "version": 1, "snapshot": 15}]
    }
    fake = _dataset_fake()
    fake.checkpoint_doc = checkpoint
    helper = make_helper(fake, dataset_name="ds_beta")
    assert helper.get_snapshot_id() == 15


# ---------------------------------------------------------------------------
# refresh_dataset
# ---------------------------------------------------------------------------

def test_refresh_dataset_returns_source(make_helper):
    """refresh_dataset() returns the _source dict from the dataset index."""
    fake = _dataset_fake()
    helper = make_helper(fake, dataset_name="ds1")
    result = helper.refresh_dataset()
    assert result == {"query": _DATASET_QUERY_SIMPLE}


def test_refresh_dataset_none_guard_on_folder_path_helper(make_helper):
    """refresh_dataset() returns {} when helper was not constructed with dataset_name."""
    fake = FakeElasticsearch()
    helper = make_helper(fake, folder_path="/ifs/data")
    assert helper.refresh_dataset() == {}


# ---------------------------------------------------------------------------
# get_all_files (restored convenience method)
# ---------------------------------------------------------------------------

def test_get_all_files_returns_added_files(make_helper):
    """get_all_files() returns all files for the scope (snapshot_id=0 full scan)."""
    pages = [[
        make_hit("/ifs/data/a.txt", lin=1, snapshot=3, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/b.txt", lin=2, snapshot=3, change_types=["ENTRY_ADDED"]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=3)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_all_files())
    assert results == [
        (Path("/ifs/data/a.txt"), 3, 1),
        (Path("/ifs/data/b.txt"), 3, 2),
    ]


def test_get_all_files_returns_modified_files(make_helper):
    """get_all_files() returns ENTRY_MODIFIED files too, unlike get_new_files()."""
    pages = [[
        make_hit("/ifs/data/a.txt", lin=1, snapshot=3, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/b.txt", lin=2, snapshot=3, change_types=["ENTRY_MODIFIED"]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=3)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_all_files())
    # Both files returned; get_all_files does not filter by change type.
    assert len(results) == 2
    paths = {str(r[0]) for r in results}
    assert "/ifs/data/a.txt" in paths
    assert "/ifs/data/b.txt" in paths


def test_get_all_files_returns_files_with_empty_change_types(make_helper):
    """get_all_files() returns files even when change_types is empty.

    get_new_files() would drop such files because 'ENTRY_ADDED' is not in [].
    get_all_files() must not apply that filter.
    """
    pages = [[
        make_hit("/ifs/data/a.txt", lin=1, snapshot=3, change_types=[]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=3)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_all_files())
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/data/a.txt"


def test_get_all_files_uses_snapshot_zero_range(make_helper):
    """get_all_files() uses snapshot_id=0, so the main-repo query has a range filter."""
    pages = [[make_hit("/ifs/data/a.txt", lin=1, snapshot=5)]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=5)
    helper = make_helper(fake, folder_path="/ifs/data")
    list(helper.get_all_files())
    paged_calls = [c for c in fake.search_calls if "aggs" not in c]
    musts = paged_calls[0]["query"]["bool"]["must"]
    assert {"range": {"metadata.snapshots.s2": {"gt": 0}}} in musts
    assert {"range": {"metadata.snapshots.s2": {"lte": 5}}} in musts


def test_get_all_files_does_not_write_checkpoint(make_helper):
    """get_all_files() must not advance the checkpoint document."""
    pages = [[make_hit("/ifs/data/a.txt", lin=1, snapshot=5)]]
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 1}]}
    fake = FakeElasticsearch(checkpoint_doc=doc, search_pages=pages, max_snapid=5)
    helper = make_helper(fake, folder_path="/ifs/data")
    list(helper.get_all_files())
    assert fake.indexed == []
