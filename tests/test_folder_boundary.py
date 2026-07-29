"""Tests for folder path boundary enforcement (U2 fix)."""

import pytest
from pathlib import Path
from tests.conftest import FakeElasticsearch


def test_folder_boundary_excludes_sibling_directories(make_helper):
    """Verify that /ifs/data/foo does not match /ifs/data/foobar/file.txt"""
    
    # Simulate ES returning both files under /ifs/data/foo and /ifs/data/foobar
    # due to match_phrase_prefix over-matching
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/file1.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foobar/file2.txt",  # Sibling directory
                            "lin": 101,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [101],
                },
            ]
        ],
        max_snapid=10,
    )
    
    helper = make_helper(fake_es, folder_path="/ifs/data/foo")
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should only get the file actually under /ifs/data/foo
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/data/foo/file1.txt"


def test_folder_boundary_excludes_prefix_match(make_helper):
    """Verify that /ifs/data does not match /ifs/database/file.txt"""
    
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/file1.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/database/file2.txt",  # Different directory
                            "lin": 101,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [101],
                },
            ]
        ],
        max_snapid=10,
    )
    
    helper = make_helper(fake_es, folder_path="/ifs/data")
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should only get the file actually under /ifs/data
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/data/file1.txt"


def test_folder_boundary_with_trailing_slash(make_helper):
    """Verify boundary check works when folder_path has trailing slash"""
    
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/file1.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foobar/file2.txt",
                            "lin": 101,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [101],
                },
            ]
        ],
        max_snapid=10,
    )
    
    # Folder path with trailing slash should be normalized
    helper = make_helper(fake_es, folder_path="/ifs/data/foo/")
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/data/foo/file1.txt"


def test_folder_boundary_accepts_nested_files(make_helper):
    """Verify that files in subdirectories are correctly included"""
    
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/file1.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/subdir/file2.txt",
                            "lin": 101,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [101],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/a/b/c/file3.txt",
                            "lin": 102,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [102],
                },
            ]
        ],
        max_snapid=10,
    )
    
    helper = make_helper(fake_es, folder_path="/ifs/data/foo")
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should get all files under /ifs/data/foo, including nested ones
    assert len(results) == 3
    paths = [str(r[0]) for r in results]
    assert "/ifs/data/foo/file1.txt" in paths
    assert "/ifs/data/foo/subdir/file2.txt" in paths
    assert "/ifs/data/foo/a/b/c/file3.txt" in paths


def test_dataset_scope_not_affected_by_folder_filter(make_helper):
    """Verify that dataset scope doesn't apply folder boundary filtering"""
    
    dataset_doc = {"query": {"match_all": {}}}
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/anywhere/file.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
            ]
        ],
        max_snapid=10,
        dataset_doc=dataset_doc,
    )
    
    # Dataset scope should not apply folder filtering
    helper = make_helper(fake_es, dataset_name="test_dataset")
    
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should get the file even though it's not under any specific folder
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/anywhere/file.txt"


def test_input_files_scope_not_affected_by_folder_filter(make_helper):
    """Verify that input_files scope doesn't apply folder boundary filtering"""
    
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/specific/file.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
            ]
        ],
        max_snapid=10,
    )
    
    # input_files scope should not apply folder filtering
    helper = make_helper(fake_es, input_files=["/ifs/specific/file.txt"])
    
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should get the file
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/specific/file.txt"


def test_folder_boundary_exact_match_accepted(make_helper):
    """Verify that a file path exactly matching the folder path is accepted (edge case)"""
    
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo",  # Exact match with folder
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
            ]
        ],
        max_snapid=10,
    )
    
    helper = make_helper(fake_es, folder_path="/ifs/data/foo")
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should accept exact match (edge case where folder itself is a file)
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/data/foo"


def test_folder_boundary_multiple_similar_paths(make_helper):
    """Verify filtering works correctly with multiple similar paths"""
    
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/test/file1.txt",
                            "lin": 100,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [100],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/test2/file2.txt",  # Similar but different
                            "lin": 101,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [101],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/testing/file3.txt",  # Similar but different
                            "lin": 102,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [102],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/test_backup/file4.txt",  # Similar but different
                            "lin": 103,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                        },
                        "metadata": {"snapshots": {"s2": 10}},
                    },
                    "sort": [103],
                },
            ]
        ],
        max_snapid=10,
    )
    
    helper = make_helper(fake_es, folder_path="/ifs/data/test")
    results = list(helper.get_directory_changes(snapshot_id=0, save_checkpoint=False))
    
    # Should only get files under /ifs/data/test, not test2, testing, or test_backup
    assert len(results) == 1
    assert str(results[0][0]) == "/ifs/data/test/file1.txt"


def test_get_all_files_applies_folder_boundary(make_helper):
    """Verify get_all_files() also applies folder boundary filtering."""
    fake_es = FakeElasticsearch(
        checkpoint_doc=None,
        search_pages=[
            [
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/file1.txt",
                            "lin": 101,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                            "file_type": "regular",
                        },
                        "metadata": {"snapshots": {"s2": 5}},
                    },
                    "sort": [101],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foobar/file2.txt",  # sibling
                            "lin": 102,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                            "file_type": "regular",
                        },
                        "metadata": {"snapshots": {"s2": 5}},
                    },
                    "sort": [102],
                },
                {
                    "_source": {
                        "data": {
                            "path": "/ifs/data/foo/sub/file3.txt",
                            "lin": 103,
                            "btime": 1000,
                            "mtime": 2000,
                            "change_types": ["ENTRY_ADDED"],
                            "file_type": "regular",
                        },
                        "metadata": {"snapshots": {"s2": 5}},
                    },
                    "sort": [103],
                },
            ],
        ],
        max_snapid=5,
    )

    helper = make_helper(fake_es, folder_path="/ifs/data/foo")

    results = list(helper.get_all_files())
    paths = [str(r[0]) for r in results]

    assert "/ifs/data/foo/file1.txt" in paths
    assert "/ifs/data/foo/sub/file3.txt" in paths
    assert "/ifs/data/foobar/file2.txt" not in paths
    assert len(results) == 2
