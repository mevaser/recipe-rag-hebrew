from __future__ import annotations

from generation import generate_answer
from retrieval import retrieve


def source_identifier(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "")
    chunk_id = chunk.get("chunk_id", "")
    if source and chunk_id:
        return f"{source} [{chunk_id}]"
    return source or chunk_id


def answer(question: str, k: int = 5) -> dict:
    retrieved_chunks = retrieve(question, k=k)
    answer_text = generate_answer(question, retrieved_chunks)
    sources = list(dict.fromkeys(source_identifier(chunk) for chunk in retrieved_chunks))
    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }
