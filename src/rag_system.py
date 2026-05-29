from __future__ import annotations

try:
    from .generation import generate_answer, get_last_context_chunks
    from .retrieval import retrieve
except ImportError:
    from generation import generate_answer, get_last_context_chunks
    from retrieval import retrieve

DEFAULT_RETRIEVAL_K = 5


def source_identifier(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "")
    chunk_id = chunk.get("chunk_id", "")
    if source and chunk_id:
        return f"{source} [{chunk_id}]"
    return source or chunk_id


def answer(
    question: str,
    prompt_version: str = "baseline",
    answer_model: str | None = None,
    generation_context_k: int | None = None,
) -> dict:
    retrieved_chunks = retrieve(question, k=DEFAULT_RETRIEVAL_K)
    answer_text = generate_answer(
        question,
        retrieved_chunks,
        prompt_version=prompt_version,
        answer_model=answer_model,
        generation_context_k=generation_context_k,
    )
    generation_chunks = get_last_context_chunks()
    sources = list(dict.fromkeys(source_identifier(chunk) for chunk in retrieved_chunks))
    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "generation_chunks": generation_chunks,
    }
