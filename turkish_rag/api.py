"""FastAPI HTTP endpoint'leri (retrieval-only)."""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "FastAPI kurulu degil: pip install fastapi uvicorn pydantic"
    ) from e

from .config import Config
from .retriever import Retriever


app = FastAPI(title="Turkish RAG", version="0.1.0")
retriever = Retriever(Config.default())


class DocIn(BaseModel):
    id: str
    text: str
    meta: dict | None = None


class IndexRequest(BaseModel):
    documents: list[DocIn]


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class ChunkOut(BaseModel):
    text: str
    meta: dict
    similarity: float


class QueryResponse(BaseModel):
    query: str
    chunks: list[ChunkOut]


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": retriever.store.count()}


@app.post("/index")
def index(req: IndexRequest):
    n = retriever.index_documents([d.model_dump() for d in req.documents])
    return {"indexed_chunks": n, "total": retriever.store.count()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    chunks = retriever.retrieve(req.query, k=req.k)
    return QueryResponse(
        query=req.query,
        chunks=[ChunkOut(text=c.text, meta=c.meta, similarity=c.similarity) for c in chunks],
    )
