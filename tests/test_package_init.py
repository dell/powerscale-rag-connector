"""Tests for the package's public surface and lazy-import machinery."""

import os
import subprocess
import sys
import types

import pytest

import powerscale_rag_connector as prc
from tests.conftest import _SRC


def test_eager_exports_available():
    # These are imported eagerly and require only elasticsearch.
    assert prc.PowerScaleHelper is not None
    assert prc.PowerScalePathLoader is not None


def test_all_contains_expected_names():
    assert set(prc.__all__) == {
        "PowerScaleDocumentLoader",
        "PowerScaleHelper",
        "PowerScalePathLoader",
        "PowerScaleUnstructuredLoader",
        "PowerScaleUnstructuredReader",
        "PowerScaleSimpleDirectoryReader",
    }


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = prc.DoesNotExist


def test_lazy_import_document_loader():
    pytest.importorskip("langchain_core")
    # Accessing the attribute triggers importlib.import_module under the hood.
    assert prc.PowerScaleDocumentLoader.__name__ == "PowerScaleDocumentLoader"


def test_lazy_import_simple_directory_reader():
    pytest.importorskip("llama_index.core")
    assert prc.PowerScaleSimpleDirectoryReader.__name__ == "PowerScaleSimpleDirectoryReader"


def test_lazy_import_fails_without_optional_dependency(monkeypatch):
    """Missing optional dependencies must surface as ModuleNotFoundError."""
    class FakeImportlib:
        def import_module(self, name, package=None):
            raise ModuleNotFoundError(f"No module named '{name}'")

    # Remove any cached lazy attribute so __getattr__ is exercised again.
    # Use delitem on the module dict so we do not trigger __getattr__
    # (and a real import) before importlib is patched.
    monkeypatch.delitem(prc.__dict__, "PowerScaleSimpleDirectoryReader", raising=False)
    monkeypatch.setattr(prc, "importlib", FakeImportlib())
    with pytest.raises(ModuleNotFoundError):
        _ = prc.PowerScaleSimpleDirectoryReader


def test_all_lazy_exports_are_listed():
    """Every lazily-imported class appears in __all__."""
    from powerscale_rag_connector import _LAZY

    for name in _LAZY:
        assert name in prc.__all__


def test_importing_package_does_not_import_optional_dependencies():
    """The package must stay importable with no optional extras installed.

    Run in a subprocess so this process's already-imported modules cannot mask
    an eager import introduced in ``__init__``.
    """
    code = (
        "import sys, os\n"
        "sys.path.insert(0, os.environ['SRC'])\n"
        "import powerscale_rag_connector\n"
        "eager = [m for m in ('langchain_core', 'langchain_unstructured',"
        " 'llama_index', 'unstructured') if m in sys.modules]\n"
        "assert not eager, f'eagerly imported: {eager}'\n"
        "print('ok')\n"
    )
    env = dict(os.environ, SRC=_SRC)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_submodule_import_does_not_shadow_class():
    """Importing a submodule must not replace the class of the same name.

    ``import powerscale_rag_connector.PowerScaleHelper`` binds the submodule as a
    package attribute under the class's name; attribute access must still yield
    the class.
    """
    import powerscale_rag_connector.PowerScaleHelper  # noqa: F401

    assert not isinstance(prc.PowerScaleHelper, types.ModuleType)
    assert prc.PowerScaleHelper.__name__ == "PowerScaleHelper"


def test_lazy_submodule_import_does_not_shadow_class():
    """Same guarantee for the lazily-imported optional classes."""
    pytest.importorskip("langchain_core")
    import powerscale_rag_connector.PowerScaleDocumentLoader  # noqa: F401

    assert not isinstance(prc.PowerScaleDocumentLoader, types.ModuleType)
    assert prc.PowerScaleDocumentLoader.__name__ == "PowerScaleDocumentLoader"
