---
title: Tiny RAG
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: TinyLlama answers questions about a company handbook with RAG
---

# Tiny RAG Space

A Gradio demo of [tiny-rag](https://github.com/didiergarcia/tiny-rag). It indexes three short
policy documents from Acme Robotics, a fictional company (handbook, expenses, security), and
answers questions with TinyLlama-1.1B-Chat, showing the retrieved chunks and their source file
next to the answer.

To deploy, create a Gradio Space and upload `app.py`, `requirements.txt` and this `README.md`.
The app downloads the documents from GitHub on startup. On the free CPU hardware the model runs
without quantization (bf16, about 2 GB), so answers take a few seconds.
