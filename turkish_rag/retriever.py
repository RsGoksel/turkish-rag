"""Retriever: indexleme + retrieval orchestrator (LLM yok)."""

from __future__ import annotations
from dataclasses import dataclass

from .chunking import chunk_text
from .config import Config
from .embedder import Embedder
from .vector_store import VectorStore


@dataclass
class Document:
    id: str
    text: str
    meta: dict | None = None


@dataclass
class RetrievedChunk:
    text: str
    meta: dict
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


class Retriever:
    """Türkçe retrieval pipeline: chunk → embed → store → query."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.embedder = Embedder(self.config.embedder)
        self.store = VectorStore(self.config.vector_store)

    def index_documents(self, docs: list[Document] | list[dict]) -> int:
        normalized = [d if isinstance(d, Document) else Document(**d) for d in docs]
        chunks, ids, metas = [], [], []
        for d in normalized:
            for j, ch in enumerate(chunk_text(d.text, self.config.chunk)):
                if not ch.strip():
                    continue
                chunks.append(ch)
                ids.append(f"{d.id}::ch{j}")
                metas.append({**(d.meta or {}), "doc_id": d.id, "chunk_idx": j})
        if not chunks:
            return 0
        embs = self.embedder.encode(chunks, show_progress=True)
        self.store.add(ids=ids, documents=chunks, embeddings=embs.tolist(), metadatas=metas)
        return len(chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.config.top_k
        q_emb = self.embedder.encode(query)[0]
        results = self.store.query(q_emb.tolist(), k=k)
        return [RetrievedChunk(**r) for r in results]
