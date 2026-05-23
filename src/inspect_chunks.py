from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect indexed chunks for manual gold set creation.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of matching chunks to print. Defaults to 20.",
    )
    parser.add_argument(
        "--category",
        help="Optional category filter. Matches substring.",
    )
    parser.add_argument(
        "--source",
        help="Optional source substring filter.",
    )
    parser.add_argument(
        "--contains",
        help="Optional text substring filter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 JSON output path for selected previews.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Indexed chunks JSON path. Defaults to data/processed/indexed_chunks.json.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {path}.")
    return chunks


def contains_filter(value: str | None, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.casefold() in (value or "").casefold()


def chunk_matches(chunk: dict, category: str | None, source: str | None, contains: str | None) -> bool:
    metadata = chunk.get("metadata", {})
    return (
        contains_filter(metadata.get("category"), category)
        and contains_filter(metadata.get("source"), source)
        and contains_filter(chunk.get("text"), contains)
    )


def select_chunks(
    chunks: list[dict],
    limit: int,
    category: str | None,
    source: str | None,
    contains: str | None,
) -> list[tuple[int, dict]]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    matches: list[tuple[int, dict]] = []
    for index, chunk in enumerate(chunks):
        if chunk_matches(chunk, category=category, source=source, contains=contains):
            matches.append((index, chunk))
            if len(matches) >= limit:
                break
    return matches


def make_preview(index: int, chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})
    text = chunk.get("text", "")
    return {
        "index": index,
        "chunk_id": chunk.get("chunk_id", ""),
        "source": metadata.get("source", ""),
        "category": metadata.get("category", ""),
        "page": metadata.get("page"),
        "word_count": len(text.split()),
        "text": text,
    }

def print_preview(preview: dict) -> None:
    print(f"Index: {preview['index']}")
    print(f"chunk_id: {preview['chunk_id']}")
    print(f"source: {preview['source']}")
    print(f"category: {preview['category']}")
    print(f"page: {preview['page']}")
    print(f"word count: {preview['word_count']}")
    print(f"text: {preview['text'][:500]}")
    print()


def save_previews(previews: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(previews, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    configure_stdout()
    args = parse_args()
    chunks = load_chunks(args.chunks_path.resolve())
    selected_chunks = select_chunks(
        chunks=chunks,
        limit=args.limit,
        category=args.category,
        source=args.source,
        contains=args.contains,
    )
    previews = [make_preview(index, chunk) for index, chunk in selected_chunks]

    print("Chunk inspection summary")
    print("========================")
    print(f"Indexed chunks loaded: {len(chunks)}")
    print(f"Matching chunks shown: {len(previews)}")
    print()

    for preview in previews:
        print_preview(preview)

    if args.output:
        output_path = args.output.resolve()
        save_previews(previews, output_path)
        print(f"Saved selected previews to: {output_path}")


if __name__ == "__main__":
    main()
