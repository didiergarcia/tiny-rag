"""Test doubles: a deterministic embedder and an LLM that records its input."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size vector. Texts that share words score higher."""

    name = "test/hash-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return out


class FakeLLM:
    name = "test/fake-llm"

    def __init__(self, reply: str = "The Philadelphia Eagles.") -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    def generate(self, messages, max_new_tokens=300):
        self.calls.append((messages, max_new_tokens))
        return self.reply


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def llm():
    return FakeLLM()


TEXTS = [
    "The Philadelphia Eagles won Super Bowl LIX against the Kansas City Chiefs.",
    "Caesars Superdome in New Orleans hosted the game on February 9.",
    "The Detroit Lions were the top seed in the NFC.",
]


@pytest.fixture
def texts():
    return list(TEXTS)
