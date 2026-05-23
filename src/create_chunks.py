from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chunking import chunk_fixed_size, chunk_full_document
from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents_dedup.json"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chunks from loaded recipe documents.")
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "full_document"],
        default="fixed_size",
        help="Chunking strategy to use. Defaults to fixed_size.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Maximum number of words per fixed-size chunk. Defaults to 300.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping words between fixed-size chunks. Defaults to 50.",
    )
    parser.add_argument(
        "--documents-path",
        "--documents",
        type=Path,
        dest="documents_path",
        default=DEFAULT_DOCUMENTS_PATH,
        help="Input documents JSON path. Defaults to data/processed/documents_dedup.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Output chunks JSON path. Defaults to data/processed/chunks.json.",
    )
    return parser.parse_args()


def load_documents(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Documents file not found: {path}. "
            "Run `python src/deduplicate_documents.py` before creating chunks."
        )
    with path.open("r", encoding="utf-8") as input_file:
        documents = json.load(input_file)
    if not isinstance(documents, list):
        raise ValueError(f"Expected a list of documents in {path}.")
    return documents


def create_chunks(
    documents: list[dict],
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    if strategy == "full_document":
        return chunk_full_document(documents)
    if strategy == "fixed_size":
        return chunk_fixed_size(documents, chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def save_chunks(chunks: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(chunks, output_file, ensure_ascii=False, indent=2)


def chunk_lengths(chunks: list[dict]) -> list[int]:
    return [int(chunk["metadata"]["chunk_size_words"]) for chunk in chunks]


def print_summary(
    documents: list[dict],
    chunks: list[dict],
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> None:
    lengths = chunk_lengths(chunks)
    average_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    print("Chunk creation summary")
    print("======================")
    print(f"Input documents: {len(documents)}")
    print(f"Selected chunking strategy: {strategy}")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap if strategy == 'fixed_size' else 'None'}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk length in words: {average_length:.2f}")
    print(f"Minimum chunk length in words: {min_length}")
    print(f"Maximum chunk length in words: {max_length}")


def print_preview(chunks: list[dict]) -> None:
    if not chunks:
        print("No chunk preview available.")
        return

    first_chunk = chunks[0]
    source = first_chunk.get("metadata", {}).get("source", "")
    preview = first_chunk.get("text", "")[:300]

    print()
    print("First chunk preview")
    print("===================")
    print(f"first chunk_id: {first_chunk['chunk_id']}")
    print(f"first chunk source: {source}")
    print(f"first 300 characters: {preview}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    documents_path = args.documents_path.resolve()
    output_path = args.output.resolve()

    documents = load_documents(documents_path)
    chunks = create_chunks(
        documents=documents,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    save_chunks(chunks, output_path)
    print_summary(
        documents=documents,
        chunks=chunks,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print_preview(chunks)
    print(f"Chunks saved to: {output_path}")


if __name__ == "__main__":
    main()
