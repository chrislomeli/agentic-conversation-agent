"""embedder.py — Local text embedding via fastembed.

Wraps fastembed.TextEmbedding to produce numpy arrays compatible with the
pgvector schema (vector(384), from sentence-transformers/all-MiniLM-L6-v2).

fastembed uses ONNX Runtime — no PyTorch or GPU required.  The model is
downloaded once on first use and cached by fastembed automatically.

Usage:
    embedder = Embedder()
    vec  = embedder.embed("some text")          # np.ndarray shape (384,)
    vecs = embedder.embed_batch(["a", "b"])     # list[np.ndarray]
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

# Must match data/schema.sql vector(N) and pg_store.EMBEDDING_DIM.
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Singleton-friendly wrapper around fastembed.TextEmbedding.

    Instantiation downloads the ONNX model on first use (~23 MB, cached).
    Share one instance across the app rather than creating per-call.
    """

    def __init__(self, model: str = _MODEL):
        self._model = TextEmbedding(model_name=model)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string; return shape-(384,) float32 array."""
        result = list(self._model.embed([text]))
        return np.array(result[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple strings in one pass; returns one array per text."""
        return [np.array(v, dtype=np.float32) for v in self._model.embed(texts)]
