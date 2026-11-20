"""Small value types shared by the index and the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A piece of a source document.

    `source` names where the text came from (a file path, `path#page=N` for a PDF
    page, or `text[i]` for the i-th string passed to `Index.from_texts`).
    `start` is the word offset of the chunk inside that source.
    """

    text: str
    source: str
    start: int


@dataclass(frozen=True)
class Hit:
    """A search result. `score` is cosine similarity: higher is closer."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """The model's answer and the chunks that were put into its prompt."""

    text: str
    sources: list[Hit] = field(default_factory=list)
