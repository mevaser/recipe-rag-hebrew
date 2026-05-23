from __future__ import annotations

import argparse
import sys

from rag_system import answer


DEFAULT_QUESTION = "איזה מתכון מתאים ללחם ללא גלוטן?"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question using the Hebrew recipe RAG pipeline.")
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to ask. Hebrew is supported.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve. Defaults to 5.",
    )
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    result = answer(args.question, k=args.k)

    print("Question")
    print("========")
    print(args.question)
    print()

    print("Answer")
    print("======")
    print(result["answer"])
    print()

    print("Sources")
    print("=======")
    for source in result["sources"]:
        print(f"- {source}")
    print()

    print("Retrieved chunk IDs")
    print("===================")
    for chunk in result["retrieved_chunks"]:
        print(f"- {chunk['chunk_id']}")


if __name__ == "__main__":
    main()
