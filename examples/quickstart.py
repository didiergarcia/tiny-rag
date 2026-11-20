"""Answer a question about the sample PDF with tiny-rag.

Run from the repo root: python examples/quickstart.py
Needs an NVIDIA GPU for 4-bit loading; drop `load_in_4bit=True` to run on CPU.
"""

from tiny_rag import RAG, HFTransformersLLM, Index

index = Index.from_files(["data/2024–25_NFL_playoffs_pg_1.pdf"], chunk_size=150)
rag = RAG(index, HFTransformersLLM(load_in_4bit=True))

answer = rag.ask("Who won the 2024 NFL Championship?", k=2)
print(answer.text)
for hit in answer.sources:
    print(f"{hit.score:.3f}  {hit.chunk.source}  {hit.chunk.text[:60]}...")
