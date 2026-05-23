from __future__ import annotations

import argparse
from pathlib import Path

from bm25_retrieval import (
    DEFAULT_BM25_INDEX_PATH,
    DEFAULT_INDEXED_CHUNKS_PATH,
    build_bm25_index,
    load_indexed_chunks,
    retrieval_text,
    save_bm25_index,
    tokenize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BM25 index over indexed recipe chunks.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_INDEXED_CHUNKS_PATH,
        help="Path to indexed_chunks.json.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_BM25_INDEX_PATH,
        help="Output BM25 pickle index path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_indexed_chunks(args.chunks_path)
    index = build_bm25_index(chunks)
    save_bm25_index(index, args.index_path)

    token_counts = [len(tokenize(retrieval_text(chunk))) for chunk in chunks]
    average_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

    print("BM25 index build summary")
    print("========================")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Average tokens per chunk: {average_tokens:.2f}")
    print(f"Input chunks path: {args.chunks_path}")
    print(f"Output BM25 index path: {args.index_path}")


if __name__ == "__main__":
    main()
