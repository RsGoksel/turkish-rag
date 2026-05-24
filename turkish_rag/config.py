"""Turkish-RAG konfigürasyonu (sadece retrieval bileşenleri)."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class EmbedderConfig:
    model_path: str = str(MODELS_DIR / "Mursit-Large-TR-Retrieval")
    device: str = "cuda"               # "cuda" yoksa otomatik "cpu"
    batch_size: int = 32
    max_seq_length: int = 2048
    normalize: bool = True              # cosine için zorunlu


@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"
    persist_dir: str = str(PROJECT_ROOT / "chroma_db")
    collection_name: str = "tr_docs"
    distance: str = "cosine"            # cosine | l2 | ip


@dataclass
class ChunkConfig:
    chunk_size: int = 400               # kelime
    chunk_overlap: int = 80             # ~%20
    split_by: str = "word"              # "word" | "sentence"


@dataclass
class Config:
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    top_k: int = 5

    @classmethod
    def default(cls) -> "Config":
        return cls()
