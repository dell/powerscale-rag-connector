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
    assert written["datasets"] == [
        {"dataset": "ds1", "version": 1, "snapshot": 5, "saved_mtime": 0}
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
    assert written["input_files"] == [
        {"paths": ["/ifs/data/a.txt"], "version": 1, "snapshot": 5, "saved_mtime": 0}
    ]
