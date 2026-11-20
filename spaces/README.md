---
title: Tiny RAG
emoji: 🏈
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: TinyLlama answers questions about one PDF page with RAG
---

# Tiny RAG Space

A Gradio demo of [tiny-rag](https://github.com/didiergarcia/tiny-rag). It indexes one page of the
Wikipedia article on the 2024–25 NFL playoffs and answers questions with TinyLlama-1.1B-Chat,
showing the retrieved chunks next to the answer.

To deploy, create a Gradio Space and upload `app.py`, `requirements.txt` and this `README.md`.
The app downloads the sample PDF from GitHub on startup. On the free CPU hardware the model runs
without quantization (bf16, about 2 GB), so answers take several seconds.

The sample text is from Wikipedia and is licensed under CC BY-SA 4.0.
