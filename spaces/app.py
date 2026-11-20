"""Gradio demo of tiny-rag over the sample PDF, for a Hugging Face Space.

Free Spaces run on CPU, so the model loads without quantization (bf16, no 4-bit) and
answers take longer than on a GPU.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import gradio as gr

from tiny_rag import RAG, HFTransformersLLM, Index

PDF_URL = (
    "https://raw.githubusercontent.com/didiergarcia/tiny-rag/main/"
    "data/2024%E2%80%9325_NFL_playoffs_pg_1.pdf"
)
PDF = Path(__file__).resolve().parent / "nfl_playoffs_2024_25_p1.pdf"

if not PDF.exists():
    urllib.request.urlretrieve(PDF_URL, PDF)

index = Index.from_files([PDF], chunk_size=150)
rag = RAG(index, HFTransformersLLM(), max_new_tokens=150)


def answer(question: str, k: float) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "Please type a question.", ""
    result = rag.ask(question, k=int(k))
    sources = "\n\n".join(
        f"**[{n}] score {hit.score:.3f}** {hit.chunk.text[:300]}…"
        for n, hit in enumerate(result.sources, start=1)
    )
    return result.text, sources


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Question", value="Who won the 2024 NFL Championship?"),
        gr.Slider(1, len(index), value=2, step=1, label="Chunks to retrieve"),
    ],
    outputs=[gr.Textbox(label="Answer"), gr.Markdown(label="Retrieved chunks")],
    title="Tiny RAG",
    description=(
        "TinyLlama-1.1B-Chat answers questions about one page of the Wikipedia article "
        "on the 2024–25 NFL playoffs, using the chunks retrieved by tiny-rag. "
        "Source: https://github.com/didiergarcia/tiny-rag"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
