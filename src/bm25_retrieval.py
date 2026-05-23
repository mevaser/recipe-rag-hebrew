from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
DEFAULT_BM25_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_index.pkl"
TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def load_indexed_chunks(chunks_path: Path = DEFAULT_INDEXED_CHUNKS_PATH) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Indexed chunks file not found: {chunks_path}")
    with chunks_path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {chunks_path}.")
    return chunks


def retrieval_text(chunk: dict) -> str:
    return str(chunk.get("indexed_text") or chunk.get("text", ""))


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(retrieval_text(chunk)) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def save_bm25_index(index: BM25Okapi, index_path: Path = DEFAULT_BM25_INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as output_file:
        pickle.dump(index, output_file)


def load_bm25_index(index_path: Path = DEFAULT_BM25_INDEX_PATH) -> BM25Okapi:
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index file not found: {index_path}. "
            "Run: python src/build_bm25_index.py"
        )
    with index_path.open("rb") as input_file:
        index = pickle.load(input_file)
    if not isinstance(index, BM25Okapi):
        raise ValueError(f"Invalid BM25 index file: {index_path}")
    return index


def validate_index_and_chunks(index: BM25Okapi, chunks: list[dict]) -> None:
    if len(index.doc_freqs) != len(chunks):
        raise ValueError(
            "BM25 index size does not match indexed chunks: "
            f"index has {len(index.doc_freqs)} documents, chunks file has {len(chunks)} chunks."
        )


def retrieve_bm25(
    query: str,
    k: int = 5,
    index_path: Path = DEFAULT_BM25_INDEX_PATH,
    chunks_path: Path = DEFAULT_INDEXED_CHUNKS_PATH,
) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    chunks = load_indexed_chunks(chunks_path)
    index = load_bm25_index(index_path)
    validate_index_and_chunks(index, chunks)

    query_tokens = tokenize(query)
    scores = index.get_scores(query_tokens)
    top_k = min(k, len(chunks))
    ranked_indices = sorted(range(len(scores)), key=lambda item: float(scores[item]), reverse=True)[:top_k]

    results: list[dict] = []
    for chunk_index in ranked_indices:
        chunk = chunks[chunk_index]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[chunk_index]),
                "metadata": chunk["metadata"],
            }
        )
    return results
