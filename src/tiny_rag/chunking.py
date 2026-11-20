"""Word-based chunking with optional overlap."""

from __future__ import annotations


def chunk_words(text: str, chunk_size: int = 150, overlap: int = 0) -> list[tuple[int, str]]:
    """Split `text` on whitespace into chunks of `chunk_size` words.

    Consecutive chunks share `overlap` words. Returns `(start_word, chunk_text)`
    pairs. Empty or whitespace-only text gives an empty list.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append((start, " ".join(words[start : start + chunk_size])))
        if start + chunk_size >= len(words):
            break
    return chunks
