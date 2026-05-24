"""Turkish RAG - Sadece retrieval (Mursit + ChromaDB).

LLM/üretim için: https://github.com/RsGoksel/Turkish-LLM-RAG
"""

from .config import Config

__version__ = "0.1.0"
__all__ = ["Config", "Embedder", "VectorStore", "Retriever"]


def __getattr__(name):
    if name == "Embedder":
        from .embedder import Embedder
        return Embedder
    if name == "VectorStore":
        from .vector_store import VectorStore
        return VectorStore
    if name == "Retriever":
        from .retriever import Retriever
        return Retriever
    raise AttributeError(f"module 'turkish_rag' has no attribute {name!r}")
