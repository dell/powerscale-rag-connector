"""Tests for folder-path boundary enforcement.

MetadataIQ paths are queried with ``match_phrase_prefix`` on an analyzed field,
which matches on token boundaries rather than string prefixes.  A scope of
``/ifs/data/foo`` therefore also matches siblings such as ``/ifs/data/foobar``,
so ``PowerScaleHelper`` post-filters every hit client-side.  These tests feed the
helper deliberately over-matched result pages and assert only in-scope paths are
yielded.
"""

import pytest

from tests.conftest import FakeElasticsearch, make_hit


def _fake_es(paths, snapshot=10):
    """Build a fake client returning one page of hits for the given paths."""
    hits = [
        make_hit(path, lin=100 + i, snapshot=snapshot, change_types=["ENTRY_ADDED"])
        for i, path in enumerate(paths)
    ]
    return FakeElasticsearch(search_pages=[hits], max_snapid=snapshot)


def _changed_paths(helper):
    return [str(r[0]) for r in helper.get_directory_changes(snapshot_id=0, save_checkpoint=False)]


@pytest.mark.parametrize(
    "folder_path, returned_paths, expected",
    [
        pytest.param(
            "/ifs/data/foo",
            ["/ifs/data/foo/file1.txt", "/ifs/data/foobar/file2.txt"],
            ["/ifs/data/foo/file1.txt"],
            id="sibling-directory-excluded",
        ),
        pytest.param(
            "/ifs/data",
            ["/ifs/data/file1.txt", "/ifs/database/file2.txt"],
            ["/ifs/data/file1.txt"],
            id="string-prefix-excluded",
        ),
        pytest.param(
            "/ifs/data/foo/",
            ["/ifs/data/foo/file1.txt", "/ifs/data/foobar/file2.txt"],
            ["/ifs/data/foo/file1.txt"],
            id="trailing-slash-normalized",
        ),
        pytest.param(
            "  /ifs/data/foo  ",
            ["/ifs/data/foo/file1.txt", "/ifs/data/foobar/file2.txt"],
            ["/ifs/data/foo/file1.txt"],
            id="surrounding-whitespace-normalized",
        ),
        pytest.param(
            "/ifs/data/foo",
            [
                "/ifs/data/foo/file1.txt",
                "/ifs/data/foo/subdir/file2.txt",
                "/ifs/data/foo/a/b/c/file3.txt",
            ],
            [
                "/ifs/data/foo/file1.txt",
                "/ifs/data/foo/subdir/file2.txt",
                "/ifs/data/foo/a/b/c/file3.txt",
            ],
            id="nested-descendants-included",
        ),
        pytest.param(
            "/ifs/data/foo",
            ["/ifs/data/foo"],
            ["/ifs/data/foo"],
            id="scope-path-itself-included",
        ),
        pytest.param(
            "/ifs/data/test",
            [
                "/ifs/data/test/file1.txt",
                "/ifs/data/test2/file2.txt",
                "/ifs/data/testing/file3.txt",
                "/ifs/data/test_backup/file4.txt",
            ],
            ["/ifs/data/test/file1.txt"],
            id="several-similar-siblings-excluded",
        ),
        pytest.param(
            "/ifs/data/foo",
            ["/ifs/data/foo.txt"],
            [],
            id="same-prefix-file-excluded",
        ),
        pytest.param(
            "/ifs/data/foo",
            ["/ifs/data", "/ifs", "/ifs/other/foo/file.txt"],
            [],
            id="ancestors-and-unrelated-excluded",
        ),
    ],
)
def test_get_directory_changes_enforces_folder_boundary(
    make_helper, folder_path, returned_paths, expected
):
    helper = make_helper(_fake_es(returned_paths), folder_path=folder_path)
    assert _changed_paths(helper) == expected


def test_get_all_files_enforces_folder_boundary(make_helper):
    """The full-scan path applies the same post-filter as the incremental one."""
    helper = make_helper(
        _fake_es(
            [
                "/ifs/data/foo/file1.txt",
                "/ifs/data/foobar/file2.txt",
                "/ifs/data/foo/sub/file3.txt",
            ],
            snapshot=5,
        ),
        folder_path="/ifs/data/foo",
    )
    paths = [str(r[0]) for r in helper.get_all_files()]
    assert paths == ["/ifs/data/foo/file1.txt", "/ifs/data/foo/sub/file3.txt"]


def test_dataset_scope_is_not_folder_filtered(make_helper):
    """A dataset scope is defined by its stored query, so no path filter applies."""
    fake_es = _fake_es(["/ifs/anywhere/file.txt"])
    fake_es.dataset_doc = {"query": {"match_all": {}}}
    helper = make_helper(fake_es, dataset_name="test_dataset")
    assert _changed_paths(helper) == ["/ifs/anywhere/file.txt"]


def test_input_files_scope_is_not_folder_filtered(make_helper):
    """An input_files scope already enumerates exact paths, so no path filter applies."""
    helper = make_helper(
        _fake_es(["/ifs/specific/file.txt"]), input_files=["/ifs/specific/file.txt"]
    )
    assert _changed_paths(helper) == ["/ifs/specific/file.txt"]
