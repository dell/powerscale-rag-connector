"""Tests for query construction, snapshot lookup and paginated search."""

import pytest

from tests.conftest import FakeElasticsearch, make_hit


# --- build_query ---------------------------------------------------------

def _musts(query):
    return query["bool"]["must"]


def _contains_key(obj, key):
    """Recursively check if *key* occurs anywhere in a dict/list structure."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_contains_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(item, key) for item in obj)
    return False


def test_build_query_folder_all_files(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    q = helper.build_query(all_files=True)
    musts = _musts(q)
    assert {"match_phrase_prefix": {"data.path": "/ifs/data"}} in musts
    assert {"term": {"data.file_type": "regular"}} in musts
    # no snapshot range filters when all_files=True
    assert not any("range" in m for m in musts)


def test_build_query_folder_incremental_adds_range(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 100
    q = helper.build_query(all_files=False, snapshot_id=5)
    musts = _musts(q)
    assert {"range": {"metadata.snapshots.s2": {"gt": 5}}} in musts
    assert {"range": {"metadata.snapshots.s2": {"lte": 100}}} in musts


def test_build_query_folder_trims_trailing_slash(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data/sub/")
    q = helper.build_query(all_files=True)
    assert {"match_phrase_prefix": {"data.path": "/ifs/data/sub"}} in _musts(q)


def test_build_query_folder_trims_trailing_space(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data/sub  ")
    q = helper.build_query(all_files=True)
    assert {"match_phrase_prefix": {"data.path": "/ifs/data/sub"}} in _musts(q)


def test_build_query_folder_trims_trailing_slash_and_space(make_helper):
    # Regression: slash followed by whitespace produced a double slash.
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data/sub/ ")
    q = helper.build_query(all_files=True)
    assert {"match_phrase_prefix": {"data.path": "/ifs/data/sub/"}} in _musts(q)


def test_build_query_folder_uses_match_phrase_prefix(make_helper):
    """Folder-path queries use match_phrase_prefix, like langchain-midx/public."""
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    q = helper.build_query(all_files=True)
    assert {"match_phrase_prefix": {"data.path": "/ifs/data"}} in _musts(q)
    assert not _contains_key(q, "max_expansions")


def test_build_query_input_files_empty_list_raises(make_helper):
    with pytest.raises(ValueError, match="cannot be empty"):
        make_helper(FakeElasticsearch(), input_files=[])


def test_build_query_input_files_should_clause(make_helper):
    helper = make_helper(
        FakeElasticsearch(), input_files=["/ifs/a.txt", "/ifs/b.txt"]
    )
    q = helper.build_query(all_files=True)
    assert q["bool"]["minimum_should_match"] == 1
    shoulds = q["bool"]["should"]
    assert {"match_phrase": {"data.path": "/ifs/a.txt"}} in shoulds
    assert {"match_phrase": {"data.path": "/ifs/b.txt"}} in shoulds


def test_build_query_dataset_uses_definition(make_helper):
    dataset_query = '{"query": {"bool": {"must": [{"term": {"data.tag": "x"}}]}}}'
    helper = make_helper(
        FakeElasticsearch(dataset_doc={"query": dataset_query}), dataset_name="ds"
    )
    q = helper.build_query(all_files=True)
    musts = _musts(q)
    assert {"term": {"data.tag": "x"}} in musts
    assert {"term": {"data.file_type": "regular"}} in musts


def test_build_query_dataset_definition_as_dict(make_helper):
    # Dataset query may be stored as a dict rather than a JSON string.
    helper = make_helper(
        FakeElasticsearch(dataset_doc={"query": {"bool": {"must": [{"term": {"data.tag": "x"}}]}}}),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    assert {"term": {"data.tag": "x"}} in _musts(q)


def test_build_query_dataset_definition_without_top_level_query_key(make_helper):
    # Tolerate dataset definitions that are the query body itself.
    helper = make_helper(
        FakeElasticsearch(dataset_doc={"query": {"bool": {"must": [{"term": {"data.tag": "y"}}]}}}),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    assert {"term": {"data.tag": "y"}} in _musts(q)


def test_build_query_dataset_match_all_wrapped_in_bool(make_helper):
    # Non-bool dataset queries must be wrapped in bool.must so the connector can
    # safely add its own filters.
    helper = make_helper(
        FakeElasticsearch(dataset_doc={"query": {"match_all": {}}}),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    assert "bool" in q
    assert "match_all" not in q
    musts = _musts(q)
    assert {"match_all": {}} in musts
    assert {"term": {"data.file_type": "regular"}} in musts


def test_build_query_dataset_term_wrapped_in_bool(make_helper):
    helper = make_helper(
        FakeElasticsearch(dataset_doc={"query": {"term": {"data.tag": "finance"}}}),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    assert "bool" in q
    assert "term" not in q
    musts = _musts(q)
    assert {"term": {"data.tag": "finance"}} in musts
    assert {"term": {"data.file_type": "regular"}} in musts


def test_build_query_dataset_full_search_body_uses_inner_query(make_helper):
    # Dataset definitions may store a full search request body; only the inner
    # "query" clause should be used.  Other top-level keys are ignored.
    dataset_query = '{"query": {"match_all": {}}, "size": 10, "aggs": {}}'
    helper = make_helper(
        FakeElasticsearch(dataset_doc={"query": dataset_query}),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    assert "bool" in q
    assert "match_all" not in q
    assert "size" not in q
    assert "aggs" not in q
    assert {"match_all": {}} in _musts(q)


def test_build_query_dataset_bool_must_as_single_dict(make_helper):
    """Dataset query with bool.must as a single dict (valid ES DSL) should normalize to list."""
    # Elasticsearch allows bool.must to be either a single clause or an array.
    # The connector must normalize to a list before appending its own filters.
    dataset_query = {"query": {"bool": {"must": {"term": {"data.tag": "x"}}}}}
    helper = make_helper(
        FakeElasticsearch(dataset_doc=dataset_query),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    musts = _musts(q)
    # Verify normalization happened: must is now a list
    assert isinstance(q["bool"]["must"], list)
    # Verify both the user's clause and the connector's filter are present
    assert {"term": {"data.tag": "x"}} in musts
    assert {"term": {"data.file_type": "regular"}} in musts


def test_build_query_dataset_bool_should_only(make_helper):
    """Dataset with only bool.should should get connector filters in bool.must."""
    # A dataset query may use only "should" clauses. The connector must add its
    # file_type filter to a new "must" array without breaking the should logic.
    dataset_query = {
        "query": {
            "bool": {
                "should": [{"term": {"data.tag": "x"}}, {"term": {"data.tag": "y"}}],
                "minimum_should_match": 1,
            }
        }
    }
    helper = make_helper(
        FakeElasticsearch(dataset_doc=dataset_query),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    # The original should clauses must be preserved
    assert "should" in q["bool"]
    assert q["bool"]["should"] == [{"term": {"data.tag": "x"}}, {"term": {"data.tag": "y"}}]
    assert q["bool"]["minimum_should_match"] == 1
    # The connector's filter must be in a new must array
    assert "must" in q["bool"]
    assert {"term": {"data.file_type": "regular"}} in q["bool"]["must"]


def test_build_query_dataset_bool_filter_only(make_helper):
    """Dataset with only bool.filter should get connector filters in bool.must."""
    # A dataset query may use only "filter" clauses. The connector must add its
    # file_type filter to a new "must" array (not to filter, to preserve scoring).
    dataset_query = {"query": {"bool": {"filter": [{"term": {"data.status": "active"}}]}}}
    helper = make_helper(
        FakeElasticsearch(dataset_doc=dataset_query),
        dataset_name="ds",
    )
    q = helper.build_query(all_files=True)
    # The original filter clauses must be preserved
    assert "filter" in q["bool"]
    assert q["bool"]["filter"] == [{"term": {"data.status": "active"}}]
    # The connector's filter must be in a new must array
    assert "must" in q["bool"]
    assert {"term": {"data.file_type": "regular"}} in q["bool"]["must"]


def test_build_query_no_scope_raises(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    # simulate a misconfigured helper with no active scope
    helper._PowerScaleHelper__folder_path = None
    helper._PowerScaleHelper__dataset_name = None
    helper._PowerScaleHelper__input_files = None
    with pytest.raises(ValueError, match="no scope configured"):
        helper.build_query(all_files=True)


# --- update_latest_snapid ------------------------------------------------

def test_update_latest_snapid_sets_value(make_helper):
    helper = make_helper(FakeElasticsearch(max_snapid=77.0), folder_path="/ifs/data")
    helper.update_latest_snapid()
    assert helper._PowerScaleHelper__latest_snapshot_id == 77


def test_update_latest_snapid_none_value_defaults_minus_one(make_helper):
    helper = make_helper(FakeElasticsearch(max_snapid=None), folder_path="/ifs/data")
    helper.update_latest_snapid()
    assert helper._PowerScaleHelper__latest_snapshot_id == -1


def test_update_latest_snapid_exception_propagates(make_helper):
    fake = FakeElasticsearch(raise_on_agg=RuntimeError("boom"))
    helper = make_helper(fake, folder_path="/ifs/data")
    with pytest.raises(RuntimeError, match="boom"):
        helper.update_latest_snapid()


def test_match_files_by_snapshot_raises_when_latest_snapid_unavailable(make_helper):
    """If the max snapshot ID cannot be determined, the caller should be told
    the scan failed, not given an empty result set because the query uses lte:-1.
    """
    fake = FakeElasticsearch(raise_on_agg=RuntimeError("es down"))
    helper = make_helper(fake, folder_path="/ifs/data")
    with pytest.raises(RuntimeError, match="es down"):
        list(helper.match_files_by_snapshot())


# --- es_search_paged -----------------------------------------------------

def test_es_search_paged_single_page(make_helper):
    pages = [[make_hit("/ifs/data/a", 1, 10), make_hit("/ifs/data/b", 2, 11)]]
    helper = make_helper(FakeElasticsearch(search_pages=pages), folder_path="/ifs/data")
    results = list(helper.es_search_paged(query={"bool": {}}))
    assert [h["_source"]["data"]["path"] for h in results] == ["/ifs/data/a", "/ifs/data/b"]


def test_es_search_paged_multiple_pages_uses_search_after(make_helper):
    pages = [
        [make_hit("/ifs/data/a", 1, 10)],
        [make_hit("/ifs/data/b", 2, 11)],
    ]
    fake = FakeElasticsearch(search_pages=pages)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.es_search_paged(query={"bool": {}}, batch_size=1))
    assert len(results) == 2
    # second search call must carry search_after from the first page's last hit
    paged_calls = [c for c in fake.search_calls if "aggs" not in c]
    assert paged_calls[1]["search_after"] == [1]


def test_es_search_paged_empty(make_helper):
    helper = make_helper(FakeElasticsearch(search_pages=[]), folder_path="/ifs/data")
    assert list(helper.es_search_paged(query={"bool": {}})) == []
