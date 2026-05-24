"""Doküman parçalama (chunking) stratejileri."""

from __future__ import annotations
import re

from .config import ChunkConfig


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, config: ChunkConfig) -> list[str]:
    """Tek metni parçalara böl."""
    if config.split_by == "word":
        return _chunk_by_words(text, config.chunk_size, config.chunk_overlap)
    if config.split_by == "sentence":
        return _chunk_by_sentences(text, config.chunk_size, config.chunk_overlap)
    raise ValueError(f"Bilinmeyen split_by: {config.split_by}")


def _chunk_by_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += step
    return out


def _chunk_by_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Cümle sınırlarında böl, sonra kelime sayısına göre grupla."""
    sentences = _SENT_SPLIT.split(text.strip())
    if not sentences:
        return []
    out: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for s in sentences:
        sw = len(s.split())
        if buf_words + sw > size and buf:
            out.append(" ".join(buf))
            # Overlap: son N kelimeyi tut
            if overlap > 0:
                tail = (" ".join(buf)).split()[-overlap:]
                buf = [" ".join(tail)]
                buf_words = len(tail)
            else:
                buf, buf_words = [], 0
        buf.append(s)
        buf_words += sw
    if buf:
        out.append(" ".join(buf))
    return out
