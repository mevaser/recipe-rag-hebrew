from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from utils import ensure_dir, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "recipes_inventory.csv"
SUPPORTED_EXTENSIONS = {".docx", ".pdf"}
CSV_COLUMNS = [
    "file_name",
    "extension",
    "relative_path",
    "parent_folder",
    "file_size_bytes",
    "last_modified",
]


def is_word_temp_file(path: Path) -> bool:
    return path.name.startswith("~$") and path.suffix.lower() == ".docx"


def iter_supported_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        logging.warning("Raw data directory does not exist: %s", raw_dir)
        return []

    files: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if is_word_temp_file(path):
            logging.info("Skipping temporary Word file: %s", path)
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        else:
            logging.info("Skipping unsupported file: %s", path)

    return sorted(files, key=lambda item: item.relative_to(raw_dir).as_posix().lower())


def build_inventory_rows(files: list[Path], raw_dir: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for path in files:
        stat = path.stat()
        relative_path = path.relative_to(raw_dir).as_posix()
        rows.append(
            {
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "relative_path": relative_path,
                "parent_folder": path.parent.name,
                "file_size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def write_inventory(rows: list[dict[str, str | int]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(files: list[Path], output_path: Path) -> None:
    counts = Counter(path.suffix.lower() for path in files)
    print("Corpus scan summary")
    print("===================")
    for extension in sorted(SUPPORTED_EXTENSIONS):
        print(f"{extension}: {counts.get(extension, 0)}")
    print(f"Total supported documents: {len(files)}")
    print(f"Inventory saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the Hebrew recipe corpus.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to scan recursively. Defaults to data/raw.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Inventory CSV output path. Defaults to data/processed/recipes_inventory.csv.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_path = args.output.resolve()

    files = iter_supported_files(raw_dir)
    rows = build_inventory_rows(files, raw_dir)
    write_inventory(rows, output_path)
    print_summary(files, output_path)


if __name__ == "__main__":
    main()
