"""Ask a question about the Acme Robotics sample documents with tiny-rag.

Run from the repo root: python examples/quickstart.py
Needs an NVIDIA GPU for 4-bit loading; drop `load_in_4bit=True` to run on CPU.
"""

from tiny_rag import RAG, HFTransformersLLM, Index

index = Index.from_files(
    ["data/acme/handbook.md", "data/acme/expenses.md", "data/acme/security.md"], chunk_size=60
)
rag = RAG(index, HFTransformersLLM(load_in_4bit=True))

answer = rag.ask("What is the daily meal limit when traveling?", k=2)
print(answer.text)
for hit in answer.sources:
    print(f"{hit.score:.3f}  {hit.chunk.source}")
