from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_loader import load_documents_with_stats
from utils import ensure_dir, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load DOCX and PDF documents from the recipe corpus.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to load recursively. Defaults to data/raw.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Documents JSON output path. Defaults to data/processed/documents.json.",
    )
    return parser.parse_args()


def save_documents(documents: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(documents, output_file, ensure_ascii=False, indent=2)


def print_summary(
    docx_documents: int,
    pdf_page_documents: int,
    total_documents: int,
    skipped_files: int,
    failed_files: int,
) -> None:
    print("Document loading summary")
    print("========================")
    print(f"DOCX documents loaded: {docx_documents}")
    print(f"PDF page documents loaded: {pdf_page_documents}")
    print(f"Total documents loaded: {total_documents}")
    print(f"Skipped files: {skipped_files}")
    print(f"Failed files: {failed_files}")


def main() -> None:
    setup_logging()
    args = parse_args()
    output_path = args.output.resolve()

    result = load_documents_with_stats(args.raw_dir.resolve())
    save_documents(result.documents, output_path)
    print_summary(
        docx_documents=result.stats.docx_documents,
        pdf_page_documents=result.stats.pdf_page_documents,
        total_documents=result.stats.total_documents,
        skipped_files=result.stats.skipped_files,
        failed_files=result.stats.failed_files,
    )
    print(f"Documents saved to: {output_path}")


if __name__ == "__main__":
    main()
