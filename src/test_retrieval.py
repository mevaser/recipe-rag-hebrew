from __future__ import annotations

import argparse
import sys

from retrieval import retrieve


DEFAULT_QUERY = "איזה מתכון כולל עוף ותפוחי אדמה?"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test retrieval over the Hebrew recipe FAISS index.")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Hebrew query to search for.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return. Defaults to 5.",
    )
    return parser.parse_args()


def print_result(rank: int, result: dict) -> None:
    metadata = result.get("metadata", {})
    preview = result.get("text", "")[:400]

    print(f"Rank: {rank}")
    print(f"Score: {result['score']:.6f}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Source: {metadata.get('source')}")
    print(f"Category: {metadata.get('category')}")
    print(f"Page: {metadata.get('page')}")
    print(f"Text preview: {preview}")
    print()


def main() -> None:
    configure_stdout()
    args = parse_args()
    results = retrieve(args.query, k=args.k)

    print("Retrieval summary")
    print("=================")
    print(f"Query: {args.query}")
    print(f"Requested k: {args.k}")
    print(f"Results returned: {len(results)}")
    print()

    for rank, result in enumerate(results, start=1):
        print_result(rank, result)


if __name__ == "__main__":
    main()
