"""Embedding models. Anything with an `embed(texts) -> array` method works."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors of shape `(len(texts), dim)`."""

    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. The model loads on first use."""

    def __init__(self, name: str = DEFAULT_EMBEDDER, device: str | None = None) -> None:
        self.name = name
        self.device = device
        self._model = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def resolve_embedder(embedder: str | Embedder) -> Embedder:
    """Accept a model name or an embedder object."""
    if isinstance(embedder, str):
        return SentenceTransformerEmbedder(embedder)
    if not isinstance(embedder, Embedder):
        raise TypeError("embedder must be a model name or have `name` and `embed(texts)`")
    return embedder
