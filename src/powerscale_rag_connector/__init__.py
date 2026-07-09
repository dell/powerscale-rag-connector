"""PowerScale RAG Connector module connects to MetadataIQ to help developers integrate PowerScale with their RAG application"""

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
        import importlib
        module = importlib.import_module(_LAZY[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module 'powerscale_rag_connector' has no attribute {name!r}")


__all__ = [
    "PowerScaleDocumentLoader",
    "PowerScaleHelper",
    "PowerScalePathLoader",
    "PowerScaleUnstructuredLoader",
    "PowerScaleUnstructuredReader",
    "PowerScaleSimpleDirectoryReader",
]
