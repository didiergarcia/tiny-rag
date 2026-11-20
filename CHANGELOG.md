# Changelog

## 0.1.0 (unreleased)

First version of the `tiny_rag` library, built from the notebook.

- `Index`: chunk text by words (with optional overlap), embed with sentence-transformers, search by cosine similarity with FAISS or NumPy, and save/load to a directory. Reads `.txt`, `.md` and `.pdf` files.
- `RAG.ask`: retrieve the top chunks, prompt the model with its chat template, and return the answer with the chunks it used. The default system prompt was chosen on practice questions (see README, Evaluation).
- LLM backends: `HFTransformersLLM` (TinyLlama by default, optional 4-bit) and `OpenAICompatibleLLM` (any `/chat/completions` server).
- pytest suite that runs on CPU without model downloads, and GitHub Actions CI for Python 3.10–3.12.
- Sample data for the library: three short policy documents from Acme Robotics, a fictional company (`data/acme/`).
- `examples/eval.py`: a 20-question evaluation over the Acme documents.
- `spaces/app.py`: a Gradio demo for a Hugging Face Space that answers questions about the Acme documents.
- The notebook moved to `examples/` and gained a section that uses the library.
