"""Gradio demo of tiny-rag over the Acme Robotics sample documents, for a Hugging Face Space.

Acme Robotics is a fictional company; its handbook, expense policy and security policy are
sample data from the tiny-rag repo. Free Spaces run on CPU, so the model loads without
quantization (bf16, no 4-bit) and answers take longer than on a GPU.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import gradio as gr

from tiny_rag import RAG, HFTransformersLLM, Index

BASE_URL = "https://raw.githubusercontent.com/didiergarcia/tiny-rag/main/data/acme/"
DOCS_DIR = Path(__file__).resolve().parent / "acme"
DOC_NAMES = ["handbook.md", "expenses.md", "security.md"]

DOCS_DIR.mkdir(exist_ok=True)
for name in DOC_NAMES:
    if not (DOCS_DIR / name).exists():
        urllib.request.urlretrieve(BASE_URL + name, DOCS_DIR / name)

index = Index.from_files([DOCS_DIR / name for name in DOC_NAMES], chunk_size=60)
rag = RAG(index, HFTransformersLLM(), max_new_tokens=150)


def answer(question: str, k: float) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "Please type a question.", ""
    result = rag.ask(question, k=int(k))
    sources = "\n\n".join(
        f"**[{n}] {Path(hit.chunk.source).name}, score {hit.score:.3f}**  \n{hit.chunk.text}"
        for n, hit in enumerate(result.sources, start=1)
    )
    return result.text, sources


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Question", value="What is the daily meal limit when traveling?"),
        gr.Slider(1, 6, value=2, step=1, label="Chunks to retrieve"),
    ],
    outputs=[gr.Textbox(label="Answer"), gr.Markdown(label="Retrieved chunks")],
    examples=[
        ["How many vacation days do full-time employees get per year?", 2],
        ["Who must approve a single expense above $10,000?", 2],
        ["How quickly must a lost laptop be reported?", 2],
    ],
    title="Tiny RAG: ask the Acme Robotics handbook",
    description=(
        "TinyLlama-1.1B-Chat answers questions about three short policy documents from "
        "Acme Robotics, a fictional company, using the chunks retrieved by tiny-rag. "
        "Check the answer against the retrieved chunks: a 1.1B model makes mistakes. "
        "Source: https://github.com/didiergarcia/tiny-rag"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
