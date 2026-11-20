from pathlib import Path

import numpy as np
import pytest

from tiny_rag import Index
from tiny_rag.index import _faiss_available

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "2024–25_NFL_playoffs_pg_1.pdf"

BACKENDS = [False] + ([True] if _faiss_available() else [])


@pytest.mark.parametrize("use_faiss", BACKENDS)
def test_search_ranks_the_matching_chunk_first(texts, embedder, use_faiss):
    index = Index.from_texts(texts, embedder=embedder, use_faiss=use_faiss)
    hits = index.search("Who won Super Bowl LIX?", k=3)
    assert hits[0].chunk.text == texts[0]
    assert hits[0].chunk.source == "text[0]"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


@pytest.mark.skipif(not _faiss_available(), reason="faiss-cpu not installed")
def test_faiss_and_numpy_agree(texts, embedder):
    with_faiss = Index.from_texts(texts, embedder=embedder, use_faiss=True)
    with_numpy = Index.from_texts(texts, embedder=embedder, use_faiss=False)
    for query in ["Super Bowl", "New Orleans February", "Lions seed"]:
        a, b = with_faiss.search(query, k=3), with_numpy.search(query, k=3)
        # Same score for every chunk; tied chunks may come back in a different order.
        score_a = {h.chunk: h.score for h in a}
        score_b = {h.chunk: h.score for h in b}
        assert score_a.keys() == score_b.keys()
        assert all(abs(score_a[c] - score_b[c]) < 1e-5 for c in score_a)


def test_k_larger_than_index_returns_everything(texts, embedder):
    index = Index.from_texts(texts, embedder=embedder)
    assert len(index.search("anything", k=10)) == len(texts)


def test_rejects_non_positive_k(texts, embedder):
    index = Index.from_texts(texts, embedder=embedder)
    with pytest.raises(ValueError):
        index.search("x", k=0)


def test_chunking_options_and_sources(embedder):
    text = " ".join(f"w{i}" for i in range(10))
    index = Index.from_texts([text], embedder=embedder, chunk_size=4, overlap=1, sources=["doc"])
    assert [c.start for c in index.chunks] == [0, 3, 6]
    assert {c.source for c in index.chunks} == {"doc"}


def test_rejects_empty_input(embedder):
    with pytest.raises(ValueError):
        Index.from_texts(["", "  "], embedder=embedder)


def test_rejects_mismatched_sources(texts, embedder):
    with pytest.raises(ValueError):
        Index.from_texts(texts, embedder=embedder, sources=["only one"])


def test_save_load_round_trip(tmp_path, texts, embedder):
    index = Index.from_texts(texts, embedder=embedder, chunk_size=5, overlap=2)
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx", embedder=embedder)
    assert loaded.chunks == index.chunks
    assert np.array_equal(loaded.vectors, index.vectors)
    query = "Caesars Superdome"
    assert loaded.search(query, k=3) == index.search(query, k=3)


def test_load_remembers_the_embedder_name(tmp_path, texts, embedder):
    Index.from_texts(texts, embedder=embedder).save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert loaded.embedder.name == embedder.name


def test_from_files_reads_txt_md_and_pdf(tmp_path, embedder):
    (tmp_path / "a.txt").write_text("The Eagles won the Super Bowl.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Notes\nThe Lions were the top seed.", encoding="utf-8")
    index = Index.from_files(
        [tmp_path / "a.txt", tmp_path / "b.md", SAMPLE_PDF], embedder=embedder, chunk_size=50
    )
    sources = {c.source for c in index.chunks}
    assert str(tmp_path / "a.txt") in sources
    assert str(tmp_path / "b.md") in sources
    assert f"{SAMPLE_PDF}#page=1" in sources
    pdf_text = " ".join(c.text for c in index.chunks if c.source.endswith("#page=1"))
    assert "Philadelphia Eagles" in pdf_text


def test_from_files_rejects_unknown_types(tmp_path, embedder):
    (tmp_path / "x.docx").write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported file type"):
        Index.from_files([tmp_path / "x.docx"], embedder=embedder)
