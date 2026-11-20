import pytest

from tiny_rag.chunking import chunk_words

WORDS = " ".join(f"w{i}" for i in range(10))


def test_chunks_without_overlap():
    chunks = chunk_words(WORDS, chunk_size=4)
    assert chunks == [(0, "w0 w1 w2 w3"), (4, "w4 w5 w6 w7"), (8, "w8 w9")]


def test_chunks_with_overlap_share_words():
    chunks = chunk_words(WORDS, chunk_size=4, overlap=2)
    assert [start for start, _ in chunks] == [0, 2, 4, 6]
    assert chunks[1][1] == "w2 w3 w4 w5"
    assert chunks[-1][1] == "w6 w7 w8 w9"


def test_text_shorter_than_one_chunk():
    assert chunk_words("only three words", chunk_size=150) == [(0, "only three words")]


def test_empty_text_gives_no_chunks():
    assert chunk_words("   \n ") == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (-1, 0), (4, 4), (4, -1)])
def test_rejects_bad_sizes(size, overlap):
    with pytest.raises(ValueError):
        chunk_words(WORDS, chunk_size=size, overlap=overlap)
