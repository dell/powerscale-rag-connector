"""PowerScale RAG Connector for integrating PowerScale MetadataIQ with RAG applications."""

import importlib
import sys
import types

from .PowerScaleHelper import PowerScaleHelper
from .PowerScalePathLoader import PowerScalePathLoader

_LAZY = {
    "PowerScaleDocumentLoader": ".PowerScaleDocumentLoader",
    "PowerScaleUnstructuredLoader": ".PowerScaleUnstructuredLoader",
    "PowerScaleUnstructuredReader": ".PowerScaleUnstructuredReader",
    "PowerScaleSimpleDirectoryReader": ".PowerScaleSimpleDirectoryReader",
}


def _resolve_lazy(name: str):
    """Import the submodule for ``name`` and cache the class on the package."""
    module = importlib.import_module(_LAZY[name], package=__name__)
    attr = getattr(module, name)
    # Importing a submodule also binds it as a package attribute under the same
    # name as the class it contains.  Overwrite that binding with the class so
    # both `import powerscale_rag_connector` and
    # `from powerscale_rag_connector import X` resolve to the class.
    sys.modules[__name__].__dict__[name] = attr
    return attr


def __getattr__(name: str):
    """PEP 562 hook for the first access to a lazily-imported class."""
    if name in _LAZY:
        return _resolve_lazy(name)
    raise AttributeError(f"module 'powerscale_rag_connector' has no attribute {name!r}")


class _LazyExportModule(types.ModuleType):
    """Module type that keeps lazy class exports from being shadowed.

    ``import powerscale_rag_connector.PowerScaleDocumentLoader`` binds the
    *submodule* to this package under the same name as the class it exports.
    That binding satisfies normal attribute lookup, so ``__getattr__`` is never
    consulted and ``powerscale_rag_connector.PowerScaleDocumentLoader`` returns
    a module instead of the class.  Intercepting attribute access here resolves
    the class instead, without importing anything until it is actually asked for.
    """

    def __getattribute__(self, name):
        if name in _LAZY:
            current = self.__dict__.get(name)
            if current is None or isinstance(current, types.ModuleType):
                return _resolve_lazy(name)
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _LazyExportModule


__all__ = [
    "PowerScaleDocumentLoader",
    "PowerScaleHelper",
    "PowerScalePathLoader",
    "PowerScaleUnstructuredLoader",
    "PowerScaleUnstructuredReader",
    "PowerScaleSimpleDirectoryReader",
]
