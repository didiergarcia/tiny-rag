# Tiny RAG

A minimal RAG example notebook (TinyLlama + FAISS + sentence-transformers) that runs in under 1 GB of GPU memory.

![TinyRAG Architecture](assets/tiny-rag.png)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/didiergarcia/tiny-rag/blob/main/tiny-rag.ipynb)

This is a learning project. One notebook asks a small model about an event after its training cutoff, sees that it doesn't know, and then fixes the answer with retrieval-augmented generation (RAG) over a one-page PDF.

## What this shows

- **4-bit quantized generation** with `transformers` and `bitsandbytes`: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) loaded in NF4 with double quantization, prompted through the model's own chat template.
- **Embeddings** with `sentence-transformers` ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).
- **A FAISS index** (`IndexFlatL2`) for finding the chunk closest to a question.
- **Small memory use.** Measured in the notebook's last cell: the 4-bit model takes 0.70 GB, and peak GPU memory for the whole notebook is 0.89 GB (RTX 5090).

## What happens in the notebook

| Step | Question | Model's answer (from the committed run) |
|---|---|---|
| No context | Who won the 2024 NFL Championship? | "The 2024 NFL Championship will be the Super Bowl, which will be held in the United States." (It doesn't know, and treats the game as a future event.) |
| Context pasted by hand | same | "…won by the Philadelphia Eagles, who defeated the defending champion Kansas City Chiefs 40-22 in the Super Bowl LIX." |
| Context found by retrieval | same | "…won by the Philadelphia Eagles, who defeated the defending champion Kansas City Chiefs 40-22 in Super Bowl LIX at Caesars Superdome in New Orleans, Louisiana." |

The retrieval steps:
1. Extract the text of the PDF's first page with PyMuPDF.
2. Split it into 150-word chunks (4 chunks for this page).
3. Embed each chunk and add the embeddings to a FAISS index.
4. Embed the question, search for the 3 nearest chunks, and put the top match into the prompt.

## Quickstart

### Colab

1. Click **Open in Colab** above.
2. Choose a GPU runtime: Runtime → Change runtime type → T4 GPU.
3. Run all cells. The notebook installs pinned packages and downloads the sample PDF itself.

### Local

You need an NVIDIA GPU with CUDA, because bitsandbytes 4-bit loading does not run on CPU.

```bash
git clone https://github.com/didiergarcia/tiny-rag.git
cd tiny-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook tiny-rag.ipynb
```

`requirements.txt` pins the versions that ran the notebook top to bottom on Ubuntu 24.04 with Python 3.12 and an RTX 5090:

```
torch==2.15.0  transformers==5.23.0  accelerate==1.16.0  bitsandbytes==0.50.2
sentence-transformers==6.3.0  faiss-cpu==1.16.0  PyMuPDF==2.1
```

## Limitations

- It is an educational demo, not a reusable package or a production system.
- It uses one document: the first page of one PDF.
- It asks one question. There is no evaluation set and no accuracy numbers.
- It retrieves 3 chunks but puts only the top one into the prompt.
- Chunking is a plain 150-word split with no overlap.
- 4-bit loading needs an NVIDIA GPU; the notebook does not run on CPU as written.
- Answers come from greedy decoding and can differ with other package versions or hardware.

## Roadmap

Possible next steps, none of which exist yet:

- Move the notebook logic into a small Python package with a typed API (`Index`, `RAG`, pluggable LLM backends).
- Add tests and CI that run on CPU.
- Add a small question set that reports retrieval hit@k and answer accuracy.
- Add a Gradio demo that can run as a Hugging Face Space.

## Project structure

```
tiny-rag/
├── README.md
├── LICENSE
├── requirements.txt
├── tiny-rag.ipynb          # The notebook
├── assets/
│   └── tiny-rag.png        # Architecture diagram
└── data/
    ├── README.md           # Source and license of the sample PDF
    └── 2024–25_NFL_playoffs_pg_1.pdf
```

## Resources

- [TinyLlama paper](https://arxiv.org/abs/2401.02385)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [bitsandbytes quantization in transformers](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

## License

The code is MIT licensed (see [LICENSE](LICENSE)). The sample PDF is Wikipedia content under CC BY-SA 4.0 (see [data/README.md](data/README.md)). PyMuPDF, which the notebook installs, is licensed under AGPL-3.0.
