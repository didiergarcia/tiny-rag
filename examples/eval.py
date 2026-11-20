"""A tiny evaluation of tiny-rag over the Acme Robotics sample documents.

Reports, for 20 hand-written questions about the three fictional policy documents in
data/acme/ (handbook, expenses, security):
- retrieval hit@1 and hit@k: an accepted answer appears in the top 1 / top k chunks
- source hit@k: a chunk from the right document is in the top k
- answer exact match: the whole answer equals an accepted answer (after normalizing)
- answer contains: an accepted answer appears in the model's answer as whole words

The documents, questions and accepted answers were written by the author.
This is a smoke test on a few small documents, not a benchmark.

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

DOCS = Path(__file__).resolve().parent.parent / "data" / "acme"

# (question, accepted answers, document that holds the answer)
QA = [
    ("How many vacation days do full-time employees get per year?", ["18", "eighteen"], "handbook"),
    (
        "How many vacation days per year do employees get after five years of service?",
        ["23", "twenty-three", "twenty three"],
        "handbook",
    ),
    (
        "How much unused vacation can be carried over to the next year?",
        ["5 days", "five days"],
        "handbook",
    ),
    ("When do carried-over vacation days expire?", ["march 31"], "handbook"),
    (
        "How many sick days does each employee get per year?",
        ["10 sick", "10 days", "ten sick", "ten days"],
        "handbook",
    ),
    (
        "When is a doctor's note required for sick leave?",
        ["3 consecutive", "three consecutive", "more than 3", "more than three"],
        "handbook",
    ),
    ("How long is paid parental leave?", ["16 weeks", "sixteen weeks"], "handbook"),
    (
        "How many days per week must hybrid employees work from the office?",
        ["2 days", "two days"],
        "handbook",
    ),
    ("How much is the home office stipend?", ["600"], "handbook"),
    ("How long is the probation period for new employees?", ["90 days", "ninety days"], "handbook"),
    ("What is the daily meal limit when traveling?", ["75"], "expenses"),
    ("What is the hotel limit per night in most cities?", ["250"], "expenses"),
    ("What is the hotel limit per night in New York?", ["350"], "expenses"),
    ("Expenses above what amount need a receipt?", ["25"], "expenses"),
    ("How soon must expense reports be submitted?", ["30 days", "thirty days"], "expenses"),
    (
        "Who must approve a single expense above $10,000?",
        ["cfo", "chief financial officer"],
        "expenses",
    ),
    ("What is the mileage reimbursement rate?", ["0.67", "67 cents"], "expenses"),
    ("What is the minimum password length?", ["14"], "security"),
    (
        "How quickly must a lost laptop be reported to IT Security?",
        ["1 hour", "one hour", "an hour"],
        "security",
    ),
    (
        "After how many minutes does a laptop screen lock?",
        ["5 minutes", "five minutes"],
        "security",
    ),
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

    files = sorted(DOCS.glob("*.md"))
    index = Index.from_files(files, chunk_size=args.chunk_size, overlap=args.overlap)
    llm = HFTransformersLLM(load_in_4bit=not args.no_4bit)
    rag = RAG(index, llm, max_new_tokens=100)

    rows = []
    started = time.perf_counter()
    for question, answers, doc in QA:
        hits = index.search(question, k=args.k)
        result = rag.ask(question, k=args.k)
        rows.append(
            {
                "question": question,
                "answer": result.text,
                "hit@1": contains(hits[0].chunk.text, answers),
                f"hit@{args.k}": any(contains(h.chunk.text, answers) for h in hits),
                f"source@{args.k}": any(Path(h.chunk.source).stem == doc for h in hits),
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
        "documents": len(files),
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
    for key in ["hit@1", f"hit@{args.k}", f"source@{args.k}", "exact", "contains"]:
        summary[key] = f"{sum(r[key] for r in rows)}/{n}"
    print("\n" + json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
