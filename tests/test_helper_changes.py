"""Tests for change iteration: match_files_by_snapshot, get_directory_changes,
get_new_files, get_deleted_files."""

from pathlib import Path

import pytest

from tests.conftest import FakeElasticsearch, make_hit


# --- match_files_by_snapshot --------------------------------------------

def test_match_files_by_snapshot_first_run_adds_range(make_helper):
    # No checkpoint => get_snapshot_id() returns -1 => main-repo style uses a
    # range filter (gt -1, lte latest_snapshot_id) which is equivalent to all files.
    pages = [[make_hit("/ifs/data/a", 1, 10)]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.match_files_by_snapshot())
    assert len(results) == 1
    paged = [c for c in fake.search_calls if "aggs" not in c][0]
    musts = paged["query"]["bool"]["must"]
    assert {"range": {"metadata.snapshots.s2": {"gt": -1}}} in musts
    assert {"range": {"metadata.snapshots.s2": {"lte": 10}}} in musts


def test_match_files_by_snapshot_incremental_adds_range(make_helper):
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5}]}
    pages = [[make_hit("/ifs/data/a", 1, 10)]]
    fake = FakeElasticsearch(checkpoint_doc=doc, search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    list(helper.match_files_by_snapshot())
    paged = [c for c in fake.search_calls if "aggs" not in c][0]
    musts = paged["query"]["bool"]["must"]
    assert {"range": {"metadata.snapshots.s2": {"gt": 5}}} in musts


# --- get_directory_changes ----------------------------------------------

def test_get_directory_changes_yields_tuples(make_helper):
    pages = [[make_hit("/ifs/data/a", 111, 10, change_types=["ENTRY_ADDED"])]]
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5}]}
    fake = FakeElasticsearch(checkpoint_doc=doc, search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_directory_changes())
    assert results == [(Path("/ifs/data/a"), 10, 111, ["ENTRY_ADDED"])]


def test_get_directory_changes_first_run_reclassifies_modified(make_helper):
    # first run (no checkpoint) => ENTRY_MODIFIED promoted to ENTRY_ADDED
    pages = [[make_hit("/ifs/data/a", 1, 10, change_types=["ENTRY_MODIFIED"])]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_directory_changes())
    assert results[0][3] == ["ENTRY_ADDED"]


def test_get_directory_changes_btime_reclassifies_modified(make_helper):
    # incremental run, but btime > saved_mtime => promote to ENTRY_ADDED
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 5, "saved_mtime": 100}
        ]
    }
    pages = [[make_hit("/ifs/data/a", 1, 10, change_types=["ENTRY_MODIFIED"], btime=200)]]
    fake = FakeElasticsearch(checkpoint_doc=doc, search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_directory_changes())
    assert results[0][3] == ["ENTRY_ADDED"]


def test_get_directory_changes_saves_checkpoint_on_completion(make_helper):
    pages = [[make_hit("/ifs/data/a", 1, 10, change_types=["ENTRY_ADDED"], mtime=321)]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    list(helper.get_directory_changes(save_checkpoint=True))
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]["folder_paths"][0]
    assert written["snapshot"] == 10
    assert written["saved_mtime"] == 321


def test_get_directory_changes_no_checkpoint_on_early_break(make_helper):
    pages = [[
        make_hit("/ifs/data/a", 1, 10, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/b", 2, 10, change_types=["ENTRY_ADDED"]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    gen = helper.get_directory_changes()
    next(gen)          # consume one
    gen.close()        # GeneratorExit before completion
    assert fake.indexed == []  # search_success never set => no checkpoint write


def test_get_directory_changes_exception_skips_checkpoint(make_helper):
    fake = FakeElasticsearch(
        search_pages=[[make_hit("/ifs/data/a", 1, 10)]],
        max_snapid=10,
        raise_on_search=RuntimeError("es down"),
    )
    helper = make_helper(fake, folder_path="/ifs/data")
    # exception is logged and re-raised to prevent partial scans from appearing complete
    with pytest.raises(RuntimeError, match="es down"):
        list(helper.get_directory_changes())
    # checkpoint should not be written when scan fails
    assert fake.indexed == []


# --- two-step incremental tests -----------------------------------------

def test_incremental_run_with_new_and_modified_files(make_helper):
    """Two-step scenario: first run returns all files; second run uses snapshot range and
    preserves ENTRY_MODIFIED for files that changed since the last checkpoint.

    MetadataIQ returns ENTRY_MODIFIED for any modified file. Reclassification to
    ENTRY_ADDED only happens on first run (no checkpoint) or when btime > saved_mtime.
    On a normal incremental run, modified files stay ENTRY_MODIFIED, so get_new_files()
    returns only genuinely new files.
    """
    # Run 1: first run, snapshot 10, one added and one modified file.
    first_pages = [[
        make_hit("/ifs/data/new.txt", 1, 10, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/modified.txt", 2, 10, change_types=["ENTRY_MODIFIED"], mtime=100),
    ]]
    fake1 = FakeElasticsearch(search_pages=first_pages, max_snapid=10)
    helper1 = make_helper(fake1, folder_path="/ifs/data")
    first_results = list(helper1.get_directory_changes(save_checkpoint=True))
    # First run reclassifies ENTRY_MODIFIED to ENTRY_ADDED.
    first_change_types = {str(p): ct for p, _s, _l, ct in first_results}
    assert first_change_types["/ifs/data/new.txt"] == ["ENTRY_ADDED"]
    assert first_change_types["/ifs/data/modified.txt"] == ["ENTRY_ADDED"]
    assert len(fake1.indexed) == 1
    written_checkpoint = fake1.indexed[0]["document"]

    # Run 2: checkpoint from run 1, snapshot 20, same modified file again plus a new file.
    second_hits = [
        make_hit("/ifs/data/another.txt", 3, 20, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/modified.txt", 2, 20, change_types=["ENTRY_MODIFIED"], mtime=200),
    ]

    # Helper 2a: get_directory_changes() on an incremental run.
    fake2 = FakeElasticsearch(
        checkpoint_doc=written_checkpoint,
        search_pages=[second_hits],
        max_snapid=20,
    )
    helper2 = make_helper(fake2, folder_path="/ifs/data")
    second_results = list(helper2.get_directory_changes(save_checkpoint=True))
    second_change_types = {str(p): ct for p, _s, _l, ct in second_results}

    # Incremental run preserves ENTRY_MODIFIED for modified files.
    assert second_change_types["/ifs/data/another.txt"] == ["ENTRY_ADDED"]
    assert second_change_types["/ifs/data/modified.txt"] == ["ENTRY_MODIFIED"]

    # The paged query must use the snapshot range (10, 20] for an incremental run.
    paged = [c for c in fake2.search_calls if "aggs" not in c][0]
    musts = paged["query"]["bool"]["must"]
    assert {"range": {"metadata.snapshots.s2": {"gt": 10}}} in musts
    assert {"range": {"metadata.snapshots.s2": {"lte": 20}}} in musts

    # Helper 2b: get_new_files() on a fresh second-run helper with the same checkpoint.
    # Calling get_new_files after get_directory_changes would advance the checkpoint
    # and query beyond snapshot 20, so use a separate helper.
    fake3 = FakeElasticsearch(
        checkpoint_doc=written_checkpoint,
        search_pages=[second_hits],
        max_snapid=20,
    )
    helper3 = make_helper(fake3, folder_path="/ifs/data")
    new_files = list(helper3.get_new_files())
    assert len(new_files) == 1
    assert str(new_files[0][0]) == "/ifs/data/another.txt"


# --- get_new_files -------------------------------------------------------

def test_get_new_files_filters_added_only(make_helper):
    pages = [[
        make_hit("/ifs/data/added", 1, 10, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/mod", 2, 10, change_types=["ENTRY_MODIFIED"]),
    ]]
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5,
                             "saved_mtime": 10_000_000}]}
    fake = FakeElasticsearch(checkpoint_doc=doc, search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_new_files())
    # only the ENTRY_ADDED file (mod has btime 0 < saved_mtime so stays MODIFIED)
    assert results == [(Path("/ifs/data/added"), 10, 1)]


def test_get_new_files_first_run_includes_reclassified_modified(make_helper):
    # First run (no checkpoint) => get_directory_changes reclassifies ENTRY_MODIFIED
    # to ENTRY_ADDED, so get_new_files() must include those files.
    pages = [[
        make_hit("/ifs/data/added", 1, 10, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/mod", 2, 10, change_types=["ENTRY_MODIFIED"]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_new_files())
    paths = {str(r[0]) for r in results}
    assert "/ifs/data/added" in paths
    assert "/ifs/data/mod" in paths  # reclassified to ENTRY_ADDED on first run


def test_empty_run_does_not_zero_saved_mtime_and_misclassify_modified(make_helper):
    """An empty successful run must not write saved_mtime=0 and then cause a
    later ENTRY_MODIFIED file to be reclassified as ENTRY_ADDED.
    """
    # First run: no files, snapshot advances to 10, saved_mtime is written as 0.
    fake1 = FakeElasticsearch(search_pages=[], max_snapid=10)
    helper1 = make_helper(fake1, folder_path="/ifs/data")
    list(helper1.get_directory_changes(save_checkpoint=True))
    assert len(fake1.indexed) == 1
    written_checkpoint = fake1.indexed[0]["document"]

    # Second run: a pre-existing file is modified. Its btime is greater than 0,
    # so the current implementation reclassifies it to ENTRY_ADDED because the
    # previous empty run wrote saved_mtime=0.
    second_pages = [[
        make_hit(
            "/ifs/data/existing.txt",
            1,
            20,
            change_types=["ENTRY_MODIFIED"],
            btime=5,
            mtime=100,
        )
    ]]
    fake2 = FakeElasticsearch(
        checkpoint_doc=written_checkpoint,
        search_pages=second_pages,
        max_snapid=20,
    )
    helper2 = make_helper(fake2, folder_path="/ifs/data")
    results = list(helper2.get_directory_changes(save_checkpoint=True))
    assert results[0][3] == ["ENTRY_MODIFIED"]


# --- scope boundary enforcement ------------------------------------------

def test_input_files_scope_rejects_descendant_and_sibling_paths(make_helper):
    """input_files uses match_phrase on the analyzed data.path field, so ES also
    returns descendants and adjacent-token paths. Only exact matches may be yielded.
    """
    requested = "/ifs/data/report"
    pages = [[
        make_hit(requested, 1, 10, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/report/inner.txt", 2, 10, change_types=["ENTRY_ADDED"]),
        make_hit("/ifs/data/report.bak", 3, 10, change_types=["ENTRY_ADDED"]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, input_files=[requested])

    paths = [str(p) for p, *_ in helper.get_directory_changes()]
    assert paths == [requested]


def test_get_all_files_input_files_scope_rejects_descendants(make_helper):
    """get_all_files() must apply the same input_files boundary as get_directory_changes."""
    requested = "/ifs/data/report"
    pages = [[
        make_hit(requested, 1, 10),
        make_hit("/ifs/data/report/inner.txt", 2, 10),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, input_files=[requested])

    paths = [str(p) for p, *_ in helper.get_all_files()]
    assert paths == [requested]


def test_input_files_scope_accepts_every_requested_path(make_helper):
    """The boundary filter must not drop legitimately requested files."""
    requested = ["/ifs/data/a.txt", "/ifs/data/nested/b.txt"]
    pages = [[
        make_hit(requested[0], 1, 10, change_types=["ENTRY_ADDED"]),
        make_hit(requested[1], 2, 10, change_types=["ENTRY_ADDED"]),
    ]]
    fake = FakeElasticsearch(search_pages=pages, max_snapid=10)
    helper = make_helper(fake, input_files=list(requested))

    paths = sorted(str(p) for p, *_ in helper.get_directory_changes())
    assert paths == sorted(requested)


# --- force_scan change-type semantics -------------------------------------

def test_force_scan_on_fresh_checkpoint_reclassifies_modified_as_added(make_helper):
    """A force scan with no saved checkpoint is still a first ingest.

    Nothing has been indexed downstream yet, so ENTRY_MODIFIED must be promoted
    to ENTRY_ADDED exactly as it is on a normal first run. Otherwise callers issue
    a delete-before-reindex for vectors that were never written.
    """
    hit = make_hit(
        "/ifs/data/a.txt", 1, 10, change_types=["ENTRY_MODIFIED"], btime=5, mtime=100
    )

    fake_default = FakeElasticsearch(search_pages=[[hit]], max_snapid=10)
    helper_default = make_helper(fake_default, folder_path="/ifs/data")
    default_run = list(helper_default.get_directory_changes())

    fake_forced = FakeElasticsearch(search_pages=[[hit]], max_snapid=10)
    helper_forced = make_helper(fake_forced, folder_path="/ifs/data")
    forced_run = list(helper_forced.get_directory_changes(snapshot_id=0))

    assert default_run[0][3] == ["ENTRY_ADDED"]
    assert forced_run[0][3] == ["ENTRY_ADDED"]


def test_force_scan_with_existing_checkpoint_keeps_modified(make_helper):
    """Once a checkpoint exists, a force scan must not fake ENTRY_ADDED."""
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 5, "saved_mtime": 50}
        ]
    }
    hit = make_hit(
        "/ifs/data/a.txt", 1, 10, change_types=["ENTRY_MODIFIED"], btime=1, mtime=100
    )
    fake = FakeElasticsearch(checkpoint_doc=doc, search_pages=[[hit]], max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")

    results = list(helper.get_directory_changes(snapshot_id=0))
    assert results[0][3] == ["ENTRY_MODIFIED"]


# --- get_deleted files ---------------------------------------------------

def test_get_deleted_files_not_implemented(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    with pytest.raises(NotImplementedError, match="ENTRY_DELETED"):
        list(helper.get_deleted_files())


def test_get_directory_changes_null_change_types(make_helper):
    """A MetadataIQ hit with data.change_types: null must not crash the iterator."""
    hit = {
        "_source": {
            "data": {
                "path": "/ifs/data/null.txt",
                "lin": 123,
                "change_types": None,
                "btime": 0,
                "mtime": 0,
            },
            "metadata": {"snapshots": {"s2": 10}},
        },
        "sort": [123],
    }
    fake = FakeElasticsearch(search_pages=[[hit]], max_snapid=10)
    helper = make_helper(fake, folder_path="/ifs/data")
    results = list(helper.get_directory_changes())
    assert results == [(Path("/ifs/data/null.txt"), 10, 123, [])]
