"""A tiny evaluation of tiny-rag over the sample PDF.

Reports, for 20 hand-written questions about the one-page sample document:
- retrieval hit@1 and hit@k: an accepted answer appears in the top 1 / top k chunks
- answer exact match: the whole answer equals an accepted answer (after normalizing)
- answer contains: an accepted answer appears in the model's answer as whole words

The questions and accepted answers were drafted from the PDF text by the author.
This is a smoke test for one small document, not a benchmark.

Run from the repo root:
    python examples/eval.py                # TinyLlama, 4-bit (NVIDIA GPU)
    python examples/eval.py --no-4bit      # no quantization, bf16 (GPU or CPU)
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

from tiny_rag import RAG, HFTransformersLLM, Index

PDF = Path(__file__).resolve().parent.parent / "data" / "2024–25_NFL_playoffs_pg_1.pdf"

QA = [
    ("Who won Super Bowl LIX?", ["philadelphia eagles", "eagles"]),
    ("Which team lost Super Bowl LIX?", ["kansas city chiefs", "chiefs"]),
    ("What was the final score of Super Bowl LIX?", ["40-22"]),
    ("In which stadium was Super Bowl LIX played?", ["caesars superdome"]),
    ("In which city was Super Bowl LIX played?", ["new orleans"]),
    ("On what date did the 2024 NFL playoffs begin?", ["january 11"]),
    ("On what date was Super Bowl LIX played?", ["february 9"]),
    ("How many teams were in the 2024-25 NFL playoffs?", ["14", "fourteen"]),
    ("How many games were played in the 2024-25 NFL playoffs?", ["13", "thirteen"]),
    ("Who were the defending champions in the 2024-25 NFL playoffs?", ["kansas city chiefs"]),
    (
        "Name one of the conference runners-up in the 2024-25 NFL playoffs.",
        ["buffalo bills", "washington commanders", "bills", "commanders"],
    ),
    (
        "How many Super Bowls have the Philadelphia Eagles won, counting Super Bowl LIX?",
        ["two", "2", "second"],
    ),
    ("What is the first round of the NFL playoffs called?", ["wild card"]),
    ("What is the second round of the NFL playoffs called?", ["divisional"]),
    (
        "Which seed receives a first-round bye?",
        ["1 seed", "seed 1", "number 1", "no 1", "first seed", "top seed", "1st seed"],
    ),
    (
        "Which seeds do the wild card teams get?",
        ["5-7", "5 to 7", "5 through 7", "5 6 and 7", "5 6 7"],
    ),
    ("Which team was the number 1 seed in the NFC?", ["detroit lions", "lions"]),
    ("Which team was the number 2 seed in the AFC?", ["buffalo bills", "bills"]),
    ("Which team was the number 4 seed in the NFC?", ["los angeles rams", "rams"]),
    ("Which team was the number 7 seed in the AFC?", ["denver broncos", "broncos"]),
]


def normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    return " ".join(text.split())


def contains(text: str, answers: list[str]) -> bool:
    text = normalize(text)
    return any(re.search(rf"(?<![\w-]){re.escape(normalize(a))}(?![\w-])", text) for a in answers)


def exact(text: str, answers: list[str]) -> bool:
    return normalize(text) in {normalize(a) for a in answers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved and given to the LLM")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--no-4bit", action="store_true", help="skip 4-bit quantization")
    parser.add_argument("--json", type=Path, help="also write per-question results here")
    args = parser.parse_args()

    index = Index.from_files([PDF], chunk_size=args.chunk_size, overlap=args.overlap)
    llm = HFTransformersLLM(load_in_4bit=not args.no_4bit)
    rag = RAG(index, llm, max_new_tokens=100)

    rows = []
    started = time.perf_counter()
    for question, answers in QA:
        hits = index.search(question, k=args.k)
        result = rag.ask(question, k=args.k)
        rows.append(
            {
                "question": question,
                "answer": result.text,
                "hit@1": contains(hits[0].chunk.text, answers),
                f"hit@{args.k}": any(contains(h.chunk.text, answers) for h in hits),
                "exact": exact(result.text, answers),
                "contains": contains(result.text, answers),
            }
        )
        r = rows[-1]
        print(
            f"[{'R' if r[f'hit@{args.k}'] else '-'}{'A' if r['contains'] else '-'}] "
            f"{question}\n      -> {result.text[:120]!r}"
        )
    elapsed = time.perf_counter() - started

    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    n = len(rows)
    summary = {
        "questions": n,
        "chunks_in_index": len(index),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "k": args.k,
        "llm": llm.name,
        "4bit": not args.no_4bit,
        "embedder": index.embedder.name,
        "device": device,
        "seconds": round(elapsed, 1),
    }
    for key in ["hit@1", f"hit@{args.k}", "exact", "contains"]:
        summary[key] = f"{sum(r[key] for r in rows)}/{n}"
    print("\n" + json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
