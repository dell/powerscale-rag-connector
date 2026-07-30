"""Tests for checkpoint read/write logic in PowerScaleHelper."""

from tests.conftest import FakeElasticsearch


# --- get_checkpoint ------------------------------------------------------

def test_get_checkpoint_found(make_helper):
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 3}]}
    helper = make_helper(FakeElasticsearch(checkpoint_doc=doc), folder_path="/ifs/data")
    found, state = helper.get_checkpoint()
    assert found is True
    assert state == doc


def test_get_checkpoint_not_found_returns_empty_root(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    found, state = helper.get_checkpoint()
    assert found is False
    assert state == {"folder_paths": [], "datasets": [], "input_files": []}


# --- get_snapshot_id -----------------------------------------------------

def _helper_with_state(make_helper, checkpoints, app_version=1):
    doc = {"folder_paths": checkpoints}
    return make_helper(
        FakeElasticsearch(checkpoint_doc=doc),
        folder_path="/ifs/data",
        app_version=app_version,
    )


def test_get_snapshot_id_match(make_helper):
    helper = _helper_with_state(
        make_helper, [{"path": "/ifs/data", "version": 1, "snapshot": 42}]
    )
    assert helper.get_snapshot_id() == 42


def test_get_snapshot_id_no_match_returns_minus_one(make_helper):
    helper = _helper_with_state(
        make_helper, [{"path": "/ifs/other", "version": 1, "snapshot": 42}]
    )
    assert helper.get_snapshot_id() == -1


def test_get_snapshot_id_version_mismatch(make_helper):
    helper = _helper_with_state(
        make_helper,
        [{"path": "/ifs/data", "version": 99, "snapshot": 42}],
        app_version=1,
    )
    assert helper.get_snapshot_id() == -1


def test_get_snapshot_id_missing_snapshot_field(make_helper):
    helper = _helper_with_state(
        make_helper,
        [{"path": "/ifs/data", "version": 1}],
    )
    assert helper.get_snapshot_id() == -1


def test_get_snapshot_id_missing_root_key(make_helper):
    # checkpoint doc exists but lacks the folder_paths root
    helper = make_helper(
        FakeElasticsearch(checkpoint_doc={"datasets": []}), folder_path="/ifs/data"
    )
    assert helper.get_snapshot_id() == -1


def test_get_snapshot_id_none_state_refreshes(make_helper):
    """Regression: None __last_state must trigger a checkpoint refresh, not crash."""
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5}]}
    helper = make_helper(FakeElasticsearch(checkpoint_doc=doc), folder_path="/ifs/data")
    helper._PowerScaleHelper__last_state = None
    # Should not raise TypeError and should reload state.
    assert helper.get_snapshot_id() == 5


# --- get_saved_mtime -----------------------------------------------------

def test_get_saved_mtime_match(make_helper):
    helper = _helper_with_state(
        make_helper,
        [{"path": "/ifs/data", "version": 1, "snapshot": 1, "saved_mtime": 1234}],
    )
    assert helper.get_saved_mtime() == 1234


def test_get_saved_mtime_missing_field_defaults_zero(make_helper):
    helper = _helper_with_state(
        make_helper, [{"path": "/ifs/data", "version": 1, "snapshot": 1}]
    )
    assert helper.get_saved_mtime() == 0


def test_get_saved_mtime_missing_root(make_helper):
    helper = make_helper(
        FakeElasticsearch(checkpoint_doc={"datasets": []}), folder_path="/ifs/data"
    )
    assert helper.get_saved_mtime() == 0


# --- init_checkpoint_doc -------------------------------------------------

def test_init_checkpoint_doc_shape(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    doc = helper.init_checkpoint_doc()
    assert doc == {"folder_paths": [], "datasets": [], "input_files": []}


# --- save_checkpoint -----------------------------------------------------

def test_save_checkpoint_skips_when_snapid_invalid(make_helper):
    fake = FakeElasticsearch()
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = -1
    helper.save_checkpoint()
    assert fake.indexed == []


def test_save_checkpoint_new_entry_written(make_helper):
    fake = FakeElasticsearch()
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 10
    helper._PowerScaleHelper__max_mtime = 555
    helper._PowerScaleHelper__files_seen = True  # simulate scan yielded files
    helper.save_checkpoint()
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]["folder_paths"]
    assert written == [
        {"path": "/ifs/data", "version": 1, "snapshot": 10, "saved_mtime": 555}
    ]


def test_save_checkpoint_updates_existing_entry(make_helper):
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 5, "saved_mtime": 1}
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper._PowerScaleHelper__max_mtime = 999
    helper._PowerScaleHelper__files_seen = True  # simulate scan yielded files
    helper.save_checkpoint()
    written = fake.indexed[0]["document"]["folder_paths"]
    assert written == [
        {"path": "/ifs/data", "version": 1, "snapshot": 20, "saved_mtime": 999}
    ]


def test_save_checkpoint_breaks_after_first_match(make_helper):
    # Duplicate entries: first snapshot=5, second snapshot=99. Latest snapshot is
    # 50, which is greater than the first but less than the second. Without break,
    # the loop would overwrite the second and then skip the write because the
    # second's stored snapshot (99) is >= 50, so the update would be lost.
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 5, "saved_mtime": 1},
            {"path": "/ifs/data", "version": 1, "snapshot": 99, "saved_mtime": 1},
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 50
    helper._PowerScaleHelper__max_mtime = 123
    helper._PowerScaleHelper__files_seen = True  # simulate scan yielded files
    helper.save_checkpoint()
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]["folder_paths"]
    assert written[0] == {
        "path": "/ifs/data", "version": 1, "snapshot": 50, "saved_mtime": 123
    }
    # The second duplicate must be left untouched.
    assert written[1] == {
        "path": "/ifs/data", "version": 1, "snapshot": 99, "saved_mtime": 1
    }


def test_save_checkpoint_skips_write_when_not_advanced(make_helper):
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 20, "saved_mtime": 1}
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    # latest equals stored snapshot => no advancement => no write
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper.save_checkpoint()
    assert fake.indexed == []


def test_save_checkpoint_defensive_none_state(make_helper):
    """If __last_state is somehow None, save_checkpoint rebuilds via init doc."""
    fake = FakeElasticsearch()
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__last_state = None
    helper._PowerScaleHelper__latest_snapshot_id = 3
    helper.save_checkpoint()
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]
    assert "folder_paths" in written and "datasets" in written


def test_save_checkpoint_adds_missing_root_keys(make_helper):
    """If a checkpoint document from a different scope is missing our root key,
    save_checkpoint adds it rather than raising KeyError."""
    # checkpoint was written by a folder_path helper and now a dataset helper reuses app_name
    fake = FakeElasticsearch(
        checkpoint_doc={"folder_paths": [{"path": "/ifs/a", "version": 1, "snapshot": 1}]},
        dataset_doc={"query": '{"query": {"bool": {"must": []}}}'},
    )
    helper = make_helper(fake, dataset_name="ds1")
    helper._PowerScaleHelper__latest_snapshot_id = 5
    helper.save_checkpoint()
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]
    assert "folder_paths" in written
    assert "datasets" in written
    # No files_seen, so saved_mtime is not written (preserves old value or omits)
    assert written["datasets"] == [
        {"dataset": "ds1", "version": 1, "snapshot": 5}
    ]


def test_save_checkpoint_adds_missing_input_files_root(make_helper):
    """Upgrading from an older checkpoint that lacks the input_files root should not
    raise KeyError when the new scope uses input_files.
    """
    fake = FakeElasticsearch(
        checkpoint_doc={"folder_paths": [{"path": "/ifs/a", "version": 1, "snapshot": 1}]},
    )
    helper = make_helper(fake, input_files=["/ifs/data/a.txt"])
    helper._PowerScaleHelper__latest_snapshot_id = 5
    helper.save_checkpoint()
    assert len(fake.indexed) == 1
    written = fake.indexed[0]["document"]
    assert "input_files" in written
    # No files_seen, so saved_mtime is not written (preserves old value or omits)
    assert written["input_files"] == [
        {"paths": ["/ifs/data/a.txt"], "version": 1, "snapshot": 5}
    ]


# --- saved_mtime preservation -------------------------------------------

def test_save_checkpoint_without_scan_preserves_saved_mtime(make_helper):
    """A bare save_checkpoint() must not zero a previously stored saved_mtime."""
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 5, "saved_mtime": 4242}
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    # get_directory_changes never ran: __files_seen is False and __max_mtime is 0.
    helper.save_checkpoint()
    written = fake.indexed[0]["document"]["folder_paths"]
    assert written == [
        {"path": "/ifs/data", "version": 1, "snapshot": 20, "saved_mtime": 4242}
    ]


def test_save_checkpoint_empty_scan_preserves_saved_mtime(make_helper):
    """A scan that matched no files must also preserve saved_mtime."""
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 5, "saved_mtime": 777}
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper._PowerScaleHelper__files_seen = False
    helper._PowerScaleHelper__max_mtime = 0
    helper.save_checkpoint()
    written = fake.indexed[0]["document"]["folder_paths"]
    assert written[0]["saved_mtime"] == 777


def test_skipped_write_leaves_in_memory_state_unchanged(make_helper):
    """When no write happens the in-memory state must match what is persisted."""
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 20, "saved_mtime": 1}
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper._PowerScaleHelper__files_seen = True
    helper._PowerScaleHelper__max_mtime = 999
    helper.save_checkpoint()
    assert fake.indexed == []
    # saved_mtime 999 was never persisted, so it must not appear in memory either.
    assert helper.get_saved_mtime() == 1


# --- trailing slash / legacy key normalization ---------------------------

def test_trailing_slash_matches_existing_checkpoint(make_helper):
    """A folder_path with a trailing slash must reuse the unslashed checkpoint."""
    doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 7, "saved_mtime": 55}
        ]
    }
    helper = make_helper(
        FakeElasticsearch(checkpoint_doc=doc), folder_path="/ifs/data/"
    )
    assert helper.get_snapshot_id() == 7
    assert helper.get_saved_mtime() == 55


def test_legacy_slashed_checkpoint_is_reused_and_normalized(make_helper):
    """A checkpoint written with a trailing slash is found and rewritten normalized."""
    doc = {
        "folder_paths": [
            {"path": "/ifs/data/", "version": 1, "snapshot": 7, "saved_mtime": 55}
        ]
    }
    fake = FakeElasticsearch(checkpoint_doc=doc)
    helper = make_helper(fake, folder_path="/ifs/data")
    assert helper.get_snapshot_id() == 7
    assert helper.get_saved_mtime() == 55

    helper._PowerScaleHelper__latest_snapshot_id = 9
    helper.save_checkpoint()
    written = fake.indexed[0]["document"]["folder_paths"]
    # Rewritten in place under the normalized key, not appended as a duplicate.
    assert written == [
        {"path": "/ifs/data", "version": 1, "snapshot": 9, "saved_mtime": 55}
    ]


# --- optimistic concurrency control -------------------------------------

def test_save_checkpoint_sends_seq_no_and_primary_term(make_helper):
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5}]}
    fake = FakeElasticsearch(checkpoint_doc=doc, seq_no=12, primary_term=3)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper.save_checkpoint()
    assert fake.indexed[0]["kwargs"] == {"if_seq_no": 12, "if_primary_term": 3}


def test_save_checkpoint_creates_new_document_with_op_type_create(make_helper):
    """A first-ever checkpoint has no version to guard, so it must require create."""
    fake = FakeElasticsearch(seq_no=None, primary_term=None)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper.save_checkpoint()
    assert fake.indexed[0]["kwargs"] == {"op_type": "create"}


def test_save_checkpoint_reloads_when_concurrent_first_run_wins(make_helper):
    """A losing first-run create must adopt the winner's document, not clobber it."""
    fake = FakeElasticsearch(seq_no=None, primary_term=None, conflicts=1)
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    # The concurrent winner's document becomes visible on reload.
    fake.checkpoint_doc = {
        "folder_paths": [
            {"path": "/ifs/data", "version": 1, "snapshot": 15, "saved_mtime": 9}
        ]
    }
    fake.seq_no = 4
    fake.primary_term = 1
    helper.save_checkpoint()
    # The retry guards on the reloaded version and keeps the winner's saved_mtime.
    # The fake advances seq_no on a conflict, mirroring the winner's write.
    assert len(fake.indexed) == 1
    assert fake.indexed[0]["kwargs"] == {"if_seq_no": 5, "if_primary_term": 1}
    assert fake.indexed[0]["document"]["folder_paths"] == [
        {"path": "/ifs/data", "version": 1, "snapshot": 20, "saved_mtime": 9}
    ]


def test_save_checkpoint_retries_on_conflict(make_helper):
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5}]}
    fake = FakeElasticsearch(
        checkpoint_doc=doc, seq_no=12, primary_term=3, conflicts=1
    )
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    helper.save_checkpoint()
    # One conflict, then a successful retry with the refreshed seq_no.
    assert len(fake.indexed) == 1
    assert fake.indexed[0]["kwargs"] == {"if_seq_no": 13, "if_primary_term": 3}


def test_save_checkpoint_gives_up_after_repeated_conflicts(make_helper, caplog):
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 5}]}
    fake = FakeElasticsearch(
        checkpoint_doc=doc, seq_no=12, primary_term=3, conflicts=99
    )
    helper = make_helper(fake, folder_path="/ifs/data")
    helper._PowerScaleHelper__latest_snapshot_id = 20
    with caplog.at_level("ERROR"):
        helper.save_checkpoint()
    assert fake.indexed == []
    assert "concurrent write conflicts" in caplog.text
