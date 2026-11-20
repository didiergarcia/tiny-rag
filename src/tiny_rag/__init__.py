"""tiny-rag: a minimal retrieval-augmented generation library."""

from .embedders import DEFAULT_EMBEDDER, Embedder, SentenceTransformerEmbedder
from .index import Index
from .llm import DEFAULT_LLM, LLM, HFTransformersLLM, OpenAICompatibleLLM
from .rag import RAG
from .types import Answer, Chunk, Hit

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_EMBEDDER",
    "DEFAULT_LLM",
    "LLM",
    "RAG",
    "Answer",
    "Chunk",
    "Embedder",
    "HFTransformersLLM",
    "Hit",
    "Index",
    "OpenAICompatibleLLM",
    "SentenceTransformerEmbedder",
]
