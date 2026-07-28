#!/usr/bin/env python3
"""Smoke test: verify installation dependencies are importable.

Run after installing the optional dependency sets you care about, e.g.:

    pip install -e ".[all]"
    python tests/installation_deps_check.py

The script treats the test-core optional dependencies as hard requirements and
prints warnings for extra/example-only packages (e.g., nv-ingest-client) that
are not installed.  It exits with code 0 if all hard-requirement imports succeed
and the public connector classes can be reached.
"""

import importlib
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import List, Tuple


# Make the src/ package importable when running the script directly.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# These are installed by tests/conftest.py and are required for the test suite.
REQUIRED_OPTIONAL_DEPS: List[Tuple[str, str]] = [
    ("elasticsearch", "elasticsearch"),
    ("langchain-core", "langchain_core"),
    ("langchain-unstructured", "langchain_unstructured"),
    ("llama-index", "llama_index"),
    ("llama-index-readers-file", "llama_index.readers.file"),
    ("unstructured", "unstructured"),
    ("fsspec", "fsspec"),
]

# These are extras that the examples may use; missing installs are warnings only.
EXTRA_OPTIONAL_DEPS: List[Tuple[str, str]] = [
    ("python-dotenv", "dotenv"),
    ("langchain-elasticsearch", "langchain_elasticsearch"),
    ("langchain-nvidia-ai-endpoints", "langchain_nvidia_ai_endpoints"),
    ("llama-index-embeddings-nvidia", "llama_index.embeddings.nvidia"),
    ("llama-index-vector-stores-elasticsearch", "llama_index.vector_stores.elasticsearch"),
    ("nv-ingest-client", "nv_ingest_client"),
]

CONNECTOR_CLASSES = [
    "PowerScaleDocumentLoader",
    "PowerScalePathLoader",
    "PowerScaleUnstructuredLoader",
    "PowerScaleUnstructuredReader",
    "PowerScaleSimpleDirectoryReader",
    "PowerScaleHelper",
]


def _dist_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"


def _check_group(deps: List[Tuple[str, str]], required: bool) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    for dist_name, module_name in deps:
        try:
            importlib.import_module(module_name)
            ver = _dist_version(dist_name)
            print(f"  OK  {dist_name:40s} ({ver:>12s})")
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            label = "FAIL" if required else "WARN"
            print(f"  {label} {dist_name:39s} {err}")
            failures.append((dist_name, err))
    return failures


def check_optional_deps() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    print("Checking required optional package imports...")
    required_failures = _check_group(REQUIRED_OPTIONAL_DEPS, required=True)

    print("\nChecking extra/example-only package imports...")
    extra_failures = _check_group(EXTRA_OPTIONAL_DEPS, required=False)

    return required_failures, extra_failures


def check_connector_classes() -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    print("\nChecking powerscale_rag_connector public classes...")
    try:
        import powerscale_rag_connector as psc
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        print(f"  FAIL powerscale_rag_connector: {err}")
        return [("powerscale_rag_connector", err)]

    for name in CONNECTOR_CLASSES:
        try:
            getattr(psc, name)
            print(f"  OK  {name}")
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {name}: {err}")
            failures.append((name, err))
    return failures


def main() -> int:
    required_failures, extra_failures = check_optional_deps()
    class_failures = check_connector_classes()

    total_required = len(REQUIRED_OPTIONAL_DEPS) + len(CONNECTOR_CLASSES)
    failed = len(required_failures) + len(class_failures)
    passed = total_required - failed

    print(f"\n{passed}/{total_required} required checks passed")
    if extra_failures:
        print(f"{len(extra_failures)} extra/example-only package(s) missing (warnings)")
    if failed:
        print(f"{failed} required failure(s); see output above.")
        return 1
    print("All required installation dependency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
