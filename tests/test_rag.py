from tiny_rag import RAG, Answer, Index


def test_ask_returns_answer_and_the_chunks_it_used(texts, embedder, llm):
    index = Index.from_texts(texts, embedder=embedder)
    answer = RAG(index, llm).ask("Who won Super Bowl LIX?", k=2)

    assert isinstance(answer, Answer)
    assert answer.text == "The Philadelphia Eagles."
    assert len(answer.sources) == 2
    assert answer.sources == index.search("Who won Super Bowl LIX?", k=2)


def test_prompt_contains_numbered_sources_and_question(texts, embedder, llm):
    index = Index.from_texts(texts, embedder=embedder)
    RAG(index, llm, max_new_tokens=50).ask("Who won Super Bowl LIX?", k=2)

    messages, max_new_tokens = llm.calls[0]
    assert max_new_tokens == 50
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "[1] (text[0]) The Philadelphia Eagles won Super Bowl LIX" in user
    assert "[2] (" in user
    assert user.endswith("Question: Who won Super Bowl LIX?")
