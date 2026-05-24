"""Embedding modeli (sentence-transformers tabanlı).

Mursit-Large-TR-Retrieval için optimize, ama herhangi bir
sentence-transformers uyumlu model ile çalışır.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import EmbedderConfig


class Embedder:
    """Sentence-transformers wrapper'ı.

    Lazy loading: model ilk encode() çağrısında yüklenir.
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        path = Path(self.config.model_path)
        if not (path / "config.json").exists() or (path / "config.json").stat().st_size == 0:
            raise FileNotFoundError(
                f"Embedder model bulunamadi: {path}\n"
                f"Indirme komutu:\n"
                f'  $env:HF_HUB_ENABLE_HF_TRANSFER = "1"\n'
                f'  hf download newmindai/Mursit-Large-TR-Retrieval --local-dir "{path}"'
            )
        from sentence_transformers import SentenceTransformer

        device = self.config.device
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self._model = SentenceTransformer(str(path), device=device)
        self._model.max_seq_length = self.config.max_seq_length

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Tek veya çoklu metin embed et. Şekil: (n, dim) float32."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            list(texts),
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity matrisi (normalize edilmiş vektörler için dot product)."""
        return a @ b.T
