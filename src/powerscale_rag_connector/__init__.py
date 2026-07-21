"""PowerScale RAG Connector for integrating PowerScale MetadataIQ with RAG applications."""

import importlib
import sys

from .PowerScaleHelper import PowerScaleHelper
from .PowerScalePathLoader import PowerScalePathLoader

_LAZY = {
    "PowerScaleDocumentLoader": ".PowerScaleDocumentLoader",
    "PowerScaleUnstructuredLoader": ".PowerScaleUnstructuredLoader",
    "PowerScaleUnstructuredReader": ".PowerScaleUnstructuredReader",
    "PowerScaleSimpleDirectoryReader": ".PowerScaleSimpleDirectoryReader",
}


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], package=__name__)
        attr = getattr(module, name)
        # importlib also exposes the submodule as a package attribute with the same
        # name as the class.  Replace that submodule attribute with the class so both
        # `import powerscale_rag_connector` and `from powerscale_rag_connector import X`
        # resolve to the class.
        setattr(sys.modules[__name__], name, attr)
        return attr
    raise AttributeError(f"module 'powerscale_rag_connector' has no attribute {name!r}")


__all__ = [
    "PowerScaleDocumentLoader",
    "PowerScaleHelper",
    "PowerScalePathLoader",
    "PowerScaleUnstructuredLoader",
    "PowerScaleUnstructuredReader",
    "PowerScaleSimpleDirectoryReader",
]
