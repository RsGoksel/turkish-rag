"""Chunking testleri (model gerektirmez)."""

from turkish_rag.chunking import chunk_text
from turkish_rag.config import ChunkConfig


def test_word_chunking_basic():
    text = "kelime " * 200
    chunks = chunk_text(text.strip(), ChunkConfig(chunk_size=50, chunk_overlap=10, split_by="word"))
    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)


def test_word_chunking_overlap():
    """Overlap dogru calismali."""
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, ChunkConfig(chunk_size=20, chunk_overlap=5, split_by="word"))
    # Step = 15, dolayisiyla son chunk en az 5 ortak kelime icermeli
    if len(chunks) >= 2:
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        assert len(first_words & second_words) >= 5


def test_sentence_chunking():
    text = "Birinci cümle. İkinci cümle. Üçüncü cümle. Dördüncü cümle."
    chunks = chunk_text(text, ChunkConfig(chunk_size=5, chunk_overlap=2, split_by="sentence"))
    assert len(chunks) >= 1


def test_empty_text():
    assert chunk_text("", ChunkConfig()) == []
    assert chunk_text("   ", ChunkConfig()) == []
