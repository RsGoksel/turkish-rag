"""Vektör veritabanı (ChromaDB tabanlı).

Abstract `VectorStore` arayüzü, farklı backend'ler (Chroma, Qdrant, FAISS)
arasında geçişi kolaylaştırır.
"""

from __future__ import annotations
from typing import Any, Sequence

from .config import VectorStoreConfig


class VectorStore:
    """ChromaDB wrapper'ı."""

    def __init__(self, config: VectorStoreConfig):
        if config.backend != "chromadb":
            raise NotImplementedError(
                f"Backend '{config.backend}' henüz desteklenmiyor. "
                f"Sadece 'chromadb' mevcut."
            )
        self.config = config
        self._client = None
        self._collection = None

    def _connect(self):
        if self._collection is not None:
            return
        import chromadb

        self._client = chromadb.PersistentClient(path=self.config.persist_dir)
        space_map = {"cosine": "cosine", "l2": "l2", "ip": "ip"}
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": space_map.get(self.config.distance, "cosine")},
        )

    @property
    def collection(self):
        self._connect()
        return self._collection

    def count(self) -> int:
        return self.collection.count()

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        """Chunk'ları indekse ekle. Aynı ID varsa upsert davranır."""
        kwargs: dict[str, Any] = dict(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(e) for e in embeddings],
        )
        if metadatas is not None:
            kwargs["metadatas"] = list(metadatas)
        # Idempotent: var olanları sil, sonra ekle
        try:
            self.collection.delete(ids=list(ids))
        except Exception:
            pass
        self.collection.add(**kwargs)

    def query(
        self,
        query_embedding: Sequence[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """En yakın k chunk'ı dön: [{text, meta, distance}, ...]"""
        kwargs: dict[str, Any] = dict(
            query_embeddings=[list(query_embedding)],
            n_results=k,
        )
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        return [
            {"text": t, "meta": m or {}, "distance": d}
            for t, m, d in zip(
                res["documents"][0],
                res["metadatas"][0] if res.get("metadatas") else [None] * len(res["documents"][0]),
                res["distances"][0],
            )
        ]

    def reset(self) -> None:
        """Koleksiyonu temizle. Geri alınamaz."""
        self._connect()
        self._client.delete_collection(self.config.collection_name)
        self._collection = None
