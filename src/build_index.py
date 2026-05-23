from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss

from embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts, load_embedding_model
from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.faiss"
DEFAULT_INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from recipe chunks.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Input chunks JSON path. Defaults to data/processed/chunks.json.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Output FAISS index path. Defaults to data/processed/index.faiss.",
    )
    parser.add_argument(
        "--indexed-chunks-path",
        type=Path,
        default=DEFAULT_INDEXED_CHUNKS_PATH,
        help="Output filtered chunks JSON path. Defaults to data/processed/indexed_chunks.json.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformers model name. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Defaults to 32.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum chunk text length in words. Defaults to 5.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {path}.")
    return chunks


def count_words(text: str) -> int:
    return len(text.split())


def filter_chunks(chunks: list[dict], min_words: int) -> list[dict]:
    if min_words < 0:
        raise ValueError("min_words must be greater than or equal to 0.")
    return [chunk for chunk in chunks if count_words(chunk.get("text", "")) >= min_words]


def retrieval_text(chunk: dict) -> str:
    return str(chunk.get("indexed_text") or chunk.get("text", ""))


def save_indexed_chunks(chunks: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(chunks, output_file, ensure_ascii=False, indent=2)


def build_faiss_index(embeddings) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Cannot build a FAISS index without embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    ensure_dir(path.parent)
    faiss.write_index(index, str(path))


def print_summary(
    chunks_loaded: int,
    chunks_indexed: int,
    model_name: str,
    embedding_dimension: int,
    index_path: Path,
    indexed_chunks_path: Path,
) -> None:
    print("Index build summary")
    print("===================")
    print(f"Chunks loaded: {chunks_loaded}")
    print(f"Chunks filtered out as too short: {chunks_loaded - chunks_indexed}")
    print(f"Chunks indexed: {chunks_indexed}")
    print(f"Embedding model name: {model_name}")
    print(f"Embedding dimension: {embedding_dimension}")
    print("FAISS index type: IndexFlatIP")
    print(f"Index output path: {index_path}")
    print(f"Indexed chunks output path: {indexed_chunks_path}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    chunks_path = args.chunks_path.resolve()
    index_path = args.index_path.resolve()
    indexed_chunks_path = args.indexed_chunks_path.resolve()

    chunks = load_chunks(chunks_path)
    indexed_chunks = filter_chunks(chunks, min_words=args.min_words)
    if not indexed_chunks:
        raise ValueError("No chunks remain after filtering; cannot build index.")

    model = load_embedding_model(args.model_name)
    texts = [retrieval_text(chunk) for chunk in indexed_chunks]
    embeddings = embed_texts(texts, model=model, batch_size=args.batch_size)
    index = build_faiss_index(embeddings)

    save_faiss_index(index, index_path)
    save_indexed_chunks(indexed_chunks, indexed_chunks_path)
    print_summary(
        chunks_loaded=len(chunks),
        chunks_indexed=len(indexed_chunks),
        model_name=args.model_name,
        embedding_dimension=embeddings.shape[1],
        index_path=index_path,
        indexed_chunks_path=indexed_chunks_path,
    )


if __name__ == "__main__":
    main()
