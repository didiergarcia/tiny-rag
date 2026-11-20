"""A small in-memory vector index over text chunks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .chunking import chunk_words
from .embedders import DEFAULT_EMBEDDER, Embedder, resolve_embedder
from .types import Chunk, Hit

_FORMAT_VERSION = 1
_TEXT_SUFFIXES = {".txt", ".md"}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


class Index:
    """Chunks plus their normalized embeddings.

    Scores are cosine similarity. Search uses FAISS (`IndexFlatIP`) when
    `faiss-cpu` is installed and a NumPy matrix product otherwise. Both are
    exact search and give the same scores; chunks with tied scores may come
    back in a different order.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        embedder: str | Embedder = DEFAULT_EMBEDDER,
        *,
        use_faiss: bool | None = None,
        _normalized: bool = False,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32) if _normalized else _normalize(vectors)
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            raise ValueError("an index needs at least one chunk")
        self.chunks = list(chunks)
        self.vectors = vectors
        self.embedder = resolve_embedder(embedder)
        self._faiss = None
        if use_faiss is None:
            use_faiss = _faiss_available()
        if use_faiss:
            import faiss

            self._faiss = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss.add(vectors)

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        embedder: str | Embedder = DEFAULT_EMBEDDER,
        chunk_size: int = 150,
        overlap: int = 0,
        sources: Sequence[str] | None = None,
        *,
        use_faiss: bool | None = None,
    ) -> Index:
        """Chunk and embed `texts`. `sources` labels each text (default `text[i]`)."""
        texts = list(texts)
        if sources is not None and len(sources) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(sources)} sources")
        chunks = []
        for i, text in enumerate(texts):
            source = sources[i] if sources is not None else f"text[{i}]"
            for start, piece in chunk_words(text, chunk_size, overlap):
                chunks.append(Chunk(text=piece, source=source, start=start))
        if not chunks:
            raise ValueError("no text to index: every input was empty")
        embedder = resolve_embedder(embedder)
        vectors = embedder.embed([c.text for c in chunks])
        return cls(chunks, vectors, embedder, use_faiss=use_faiss)

    @classmethod
    def from_files(
        cls,
        paths: Iterable[str | Path],
        embedder: str | Embedder = DEFAULT_EMBEDDER,
        chunk_size: int = 150,
        overlap: int = 0,
        *,
        use_faiss: bool | None = None,
    ) -> Index:
        """Index .txt, .md and .pdf files. Each PDF page is its own source (`path#page=N`)."""
        texts, sources = [], []
        for path in map(Path, paths):
            suffix = path.suffix.lower()
            if suffix in _TEXT_SUFFIXES:
                texts.append(path.read_text(encoding="utf-8"))
                sources.append(str(path))
            elif suffix == ".pdf":
                from pypdf import PdfReader

                for number, page in enumerate(PdfReader(path).pages, start=1):
                    texts.append(page.extract_text() or "")
                    sources.append(f"{path}#page={number}")
            else:
                raise ValueError(
                    f"unsupported file type {suffix!r} for {path} (use .txt, .md or .pdf)"
                )
        return cls.from_texts(
            texts, embedder, chunk_size, overlap, sources=sources, use_faiss=use_faiss
        )

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """Return up to `k` chunks, most similar first."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k = min(k, len(self.chunks))
        q = _normalize(self.embedder.embed([query]))
        if q.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"query vector has {q.shape[1]} dims but the index has {self.vectors.shape[1]}"
            )
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            pairs = zip(ids[0].tolist(), scores[0].tolist(), strict=True)
        else:
            scores = self.vectors @ q[0]
            # Stable sort: equal scores keep insertion order.
            order = np.argsort(-scores, kind="stable")[:k]
            pairs = ((int(i), float(scores[i])) for i in order)
        return [Hit(chunk=self.chunks[i], score=float(s)) for i, s in pairs]

    def save(self, path: str | Path) -> None:
        """Write the index to a directory (`chunks.json` and `vectors.npy`)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": _FORMAT_VERSION,
            "embedder": self.embedder.name,
            "chunks": [{"text": c.text, "source": c.source, "start": c.start} for c in self.chunks],
        }
        (path / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        np.save(path / "vectors.npy", self.vectors)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: str | Embedder | None = None,
        *,
        use_faiss: bool | None = None,
    ) -> Index:
        """Read an index written by `save`. By default, reuse the embedder it was built with."""
        path = Path(path)
        meta = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        if meta.get("format") != _FORMAT_VERSION:
            raise ValueError(f"unsupported index format {meta.get('format')!r} in {path}")
        chunks = [Chunk(**c) for c in meta["chunks"]]
        vectors = np.load(path / "vectors.npy")
        # The saved vectors are already normalized; re-normalizing would change their bits.
        return cls(
            chunks, vectors, embedder or meta["embedder"], use_faiss=use_faiss, _normalized=True
        )
