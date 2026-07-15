"""PowerScale RAG Connector module connects to MetadataIQ to help developers integrate PowerScale with their RAG application"""

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
        # importlib binds the submodule (whose name collides with the class name)
        # onto this package; overwrite that binding with the class so both
        # `import powerscale_rag_connector as p; p.X` and
        # `from powerscale_rag_connector import X` return the class, not the module.
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
