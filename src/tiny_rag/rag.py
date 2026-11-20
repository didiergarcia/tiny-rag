"""Retrieval-augmented question answering."""

from __future__ import annotations

from .index import Index
from .llm import LLM
from .types import Answer, Hit

# "If the context does not contain the answer, say you don't know." made TinyLlama open 11 of
# 20 answerable eval questions with "The context does not contain the answer"; this wording
# scored better on held-out practice questions (7/10 vs 5/10). See README, Evaluation.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context. "
    "Check the context carefully before saying that you don't know."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Number each retrieved chunk so the answer can be checked against its sources."""
    context = "\n\n".join(
        f"[{n}] ({hit.chunk.source}) {hit.chunk.text}" for n, hit in enumerate(hits, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


class RAG:
    """Search the index, put the top chunks into the prompt, and ask the LLM."""

    def __init__(
        self,
        index: Index,
        llm: LLM,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 300,
    ) -> None:
        self.index = index
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    def ask(self, question: str, k: int = 4) -> Answer:
        """Return the model's answer and the `k` chunks it was given."""
        hits = self.index.search(question, k)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_prompt(question, hits)},
        ]
        text = self.llm.generate(messages, max_new_tokens=self.max_new_tokens)
        return Answer(text=text, sources=hits)
