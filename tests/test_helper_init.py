"""Tests for PowerScaleHelper.__init__ scope validation and configuration."""

import pytest

from tests.conftest import FakeElasticsearch


# --- invalid scope combinations -----------------------------------------

def test_no_scope_raises(make_helper):
    with pytest.raises(ValueError, match="select one of"):
        make_helper(FakeElasticsearch())


def test_input_files_none_raises(make_helper):
    with pytest.raises(ValueError, match="select one of"):
        make_helper(FakeElasticsearch(), input_files=None)


def test_folder_path_and_input_files_mutually_exclusive(make_helper):
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_helper(
            FakeElasticsearch(),
            folder_path="/ifs/data",
            input_files=["/ifs/data/a.txt"],
        )


def test_dataset_and_folder_path_mutually_exclusive(make_helper):
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_helper(
            FakeElasticsearch(dataset_doc={"query": '{"query": {}}'}),
            dataset_name="ds",
            folder_path="/ifs/data",
        )


def test_folder_path_must_start_with_ifs(make_helper):
    with pytest.raises(ValueError, match="must start with '/ifs'"):
        make_helper(FakeElasticsearch(), folder_path="/mnt/data")


def test_input_files_must_be_list_of_str(make_helper):
    with pytest.raises(ValueError, match="must be a List"):
        make_helper(FakeElasticsearch(), input_files="/ifs/a.txt")


def test_input_files_non_str_element(make_helper):
    with pytest.raises(ValueError, match="must be a List"):
        make_helper(FakeElasticsearch(), input_files=["/ifs/a.txt", 5])


def test_input_files_must_start_with_ifs(make_helper):
    with pytest.raises(ValueError, match="must start with '/ifs'"):
        make_helper(FakeElasticsearch(), input_files=["/ifs/a.txt", "/tmp/b.txt"])


def test_input_files_empty_list_raises(make_helper):
    with pytest.raises(ValueError, match="cannot be empty"):
        make_helper(FakeElasticsearch(), input_files=[])


# --- valid scope configuration ------------------------------------------

def test_folder_path_scope_config(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    assert helper._PowerScaleHelper__checkpoint_root == "folder_paths"
    assert helper._PowerScaleHelper__checkpoint_key == "path"
    assert helper._PowerScaleHelper__checkpoint_value == "/ifs/data"


def test_dataset_scope_config_calls_refresh(make_helper):
    fake = FakeElasticsearch(dataset_doc={"query": '{"query": {"match_all": {}}}'})
    helper = make_helper(fake, dataset_name="mydataset")
    assert helper._PowerScaleHelper__checkpoint_root == "datasets"
    assert helper._PowerScaleHelper__checkpoint_key == "dataset"
    assert helper._PowerScaleHelper__checkpoint_value == "mydataset"
    # refresh_dataset performed a get against the dataset index
    assert (FakeElasticsearch.DATASET_INDEX, "mydataset") in fake.get_calls


def test_input_files_scope_is_sorted(make_helper):
    helper = make_helper(
        FakeElasticsearch(),
        input_files=["/ifs/b.txt", "/ifs/a.txt"],
    )
    assert helper._PowerScaleHelper__checkpoint_root == "input_files"
    assert helper._PowerScaleHelper__checkpoint_key == "paths"
    # normalized (sorted) for stable matching
    assert helper._PowerScaleHelper__checkpoint_value == ["/ifs/a.txt", "/ifs/b.txt"]


def test_document_name_defaults_to_app_name(make_helper):
    helper = make_helper(
        FakeElasticsearch(), folder_path="/ifs/data", app_name="my_app"
    )
    assert helper._PowerScaleHelper__document_name == "my_app"


def test_init_loads_existing_checkpoint(make_helper):
    doc = {"folder_paths": [{"path": "/ifs/data", "version": 1, "snapshot": 7}]}
    helper = make_helper(FakeElasticsearch(checkpoint_doc=doc), folder_path="/ifs/data")
    assert helper._PowerScaleHelper__last_state == doc


def test_init_first_run_empty_state(make_helper):
    helper = make_helper(FakeElasticsearch(), folder_path="/ifs/data")
    assert helper._PowerScaleHelper__last_state == {
        "folder_paths": [],
        "datasets": [],
        "input_files": [],
    }
