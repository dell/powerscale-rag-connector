"""Tests for the package's public surface and lazy-import machinery."""

import pytest

import powerscale_rag_connector as prc


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
        prc.DoesNotExist


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
        prc.PowerScaleSimpleDirectoryReader


def test_all_lazy_exports_are_listed():
    """Every lazily-imported class appears in __all__."""
    from powerscale_rag_connector import _LAZY

    for name in _LAZY:
        assert name in prc.__all__
