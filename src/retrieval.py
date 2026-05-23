from __future__ import annotations

import json
from pathlib import Path

import faiss

try:
    from .embeddings import DEFAULT_EMBEDDING_MODEL, embed_query, load_embedding_model
except ImportError:
    from embeddings import DEFAULT_EMBEDDING_MODEL, embed_query, load_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.faiss"
DEFAULT_INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"


def load_faiss_index(index_path: Path = DEFAULT_INDEX_PATH) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_indexed_chunks(indexed_chunks_path: Path = DEFAULT_INDEXED_CHUNKS_PATH) -> list[dict]:
    if not indexed_chunks_path.exists():
        raise FileNotFoundError(f"Indexed chunks file not found: {indexed_chunks_path}")

    with indexed_chunks_path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {indexed_chunks_path}.")
    return chunks


def validate_index_and_chunks(index: faiss.Index, chunks: list[dict]) -> None:
    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index size does not match indexed chunks: "
            f"index has {index.ntotal} vectors, chunks file has {len(chunks)} chunks."
        )


def retrieve(query: str, k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    index = load_faiss_index()
    chunks = load_indexed_chunks()
    validate_index_and_chunks(index, chunks)

    top_k = min(k, index.ntotal)
    model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    query_embedding = embed_query(query, model=model).reshape(1, -1)
    scores, indices = index.search(query_embedding, top_k)

    results: list[dict] = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue
        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
                "metadata": chunk["metadata"],
            }
        )

    return results
