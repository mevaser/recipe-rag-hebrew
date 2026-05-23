from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents_dedup.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "dedup_report.json"
DEFAULT_THRESHOLD = 0.92
PUNCTUATION_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


@dataclass
class PreparedDocument:
    index: int
    document: dict
    normalized_text: str
    content_hash: str
    token_set: set[str]
    word_count: int


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate loaded recipe documents.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input documents JSON path. Defaults to data/processed/documents.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output deduplicated documents JSON path. Defaults to data/processed/documents_dedup.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output deduplication report JSON path. Defaults to data/processed/dedup_report.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Near-duplicate token Jaccard threshold. Defaults to 0.92.",
    )
    return parser.parse_args()


def load_documents(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        documents = json.load(input_file)
    if not isinstance(documents, list):
        raise ValueError(f"Expected a list of documents in {path}.")
    return documents


def normalize_text_for_dedup(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCTUATION_TRANSLATION)
    text = re.sub(r"[^\w\s\u0590-\u05ff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_documents(documents: list[dict]) -> list[PreparedDocument]:
    prepared_documents: list[PreparedDocument] = []
    for index, document in enumerate(documents):
        normalized_text = normalize_text_for_dedup(document.get("text", ""))
        token_set = set(normalized_text.split())
        content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        prepared_documents.append(
            PreparedDocument(
                index=index,
                document=document,
                normalized_text=normalized_text,
                content_hash=content_hash,
                token_set=token_set,
                word_count=len(document.get("text", "").split()),
            )
        )
    return prepared_documents


def token_jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def representative_sort_key(prepared_document: PreparedDocument) -> tuple[int, int, int, str]:
    metadata = prepared_document.document.get("metadata", {})
    file_type = metadata.get("file_type", "")
    relative_path = metadata.get("relative_path", "")
    docx_priority = 0 if file_type == "docx" else 1
    return (docx_priority, -prepared_document.word_count, len(relative_path), relative_path)


def source_path(document: dict) -> str:
    return str(document.get("metadata", {}).get("source", "")).strip()


def source_file_type(document: dict) -> str:
    metadata = document.get("metadata", {})
    file_type = str(metadata.get("file_type", "")).casefold().strip().lstrip(".")
    if file_type:
        return file_type
    return Path(source_path(document)).suffix.casefold().strip().lstrip(".")


def normalized_filename_stem(document: dict) -> str:
    source = source_path(document)
    if not source:
        return ""
    return Path(source).stem.strip().casefold()


def choose_representative(group: list[int], prepared_documents: list[PreparedDocument]) -> int:
    return min(group, key=lambda index: representative_sort_key(prepared_documents[index]))


def choose_filename_stem_representative(group: list[int], documents: list[dict]) -> int:
    for index in group:
        if source_file_type(documents[index]) == "docx":
            return index
    return group[0]


def detect_duplicate_groups(prepared_documents: list[PreparedDocument], threshold: float) -> tuple[list[list[int]], dict[tuple[int, int], float]]:
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1.")

    union_find = UnionFind(len(prepared_documents))
    pair_scores: dict[tuple[int, int], float] = {}
    hash_to_indices: dict[str, list[int]] = {}

    for document in prepared_documents:
        hash_to_indices.setdefault(document.content_hash, []).append(document.index)

    for indices in hash_to_indices.values():
        if len(indices) < 2:
            continue
        first = indices[0]
        for duplicate in indices[1:]:
            union_find.union(first, duplicate)
            pair_scores[(min(first, duplicate), max(first, duplicate))] = 1.0

    for left_index, left_document in enumerate(prepared_documents):
        for right_index in range(left_index + 1, len(prepared_documents)):
            if union_find.find(left_index) == union_find.find(right_index):
                continue
            right_document = prepared_documents[right_index]
            similarity = token_jaccard_similarity(left_document.token_set, right_document.token_set)
            if similarity >= threshold:
                union_find.union(left_index, right_index)
                pair_scores[(left_index, right_index)] = similarity

    grouped: dict[int, list[int]] = {}
    for index in range(len(prepared_documents)):
        grouped.setdefault(union_find.find(index), []).append(index)

    duplicate_groups = [indices for indices in grouped.values() if len(indices) > 1]
    return duplicate_groups, pair_scores


def similarity_to_representative(kept_index: int, removed_index: int, prepared_documents: list[PreparedDocument], pair_scores: dict[tuple[int, int], float]) -> float:
    key = (min(kept_index, removed_index), max(kept_index, removed_index))
    if key in pair_scores:
        return pair_scores[key]
    return token_jaccard_similarity(
        prepared_documents[kept_index].token_set,
        prepared_documents[removed_index].token_set,
    )


def build_duplicate_group_report(
    group: list[int],
    prepared_documents: list[PreparedDocument],
    pair_scores: dict[tuple[int, int], float],
) -> tuple[dict, int, list[int]]:
    kept_index = choose_representative(group, prepared_documents)
    removed_indices = [index for index in group if index != kept_index]
    kept_document = prepared_documents[kept_index].document
    removed_documents = [prepared_documents[index].document for index in removed_indices]
    similarities = [
        similarity_to_representative(kept_index, removed_index, prepared_documents, pair_scores)
        for removed_index in removed_indices
    ]

    report = {
        "kept_doc_id": kept_document.get("doc_id", ""),
        "removed_doc_ids": [document.get("doc_id", "") for document in removed_documents],
        "kept_source": kept_document.get("metadata", {}).get("source", ""),
        "removed_sources": [document.get("metadata", {}).get("source", "") for document in removed_documents],
        "similarity": max(similarities) if similarities else 1.0,
    }
    return report, kept_index, removed_indices


def deduplicate_by_filename_stem(documents: list[dict]) -> tuple[list[dict], list[dict], set[int]]:
    stem_to_indices: dict[str, list[int]] = {}
    for index, document in enumerate(documents):
        stem = normalized_filename_stem(document)
        if stem:
            stem_to_indices.setdefault(stem, []).append(index)

    removed_indices: set[int] = set()
    filename_stem_reports: list[dict] = []

    for stem, group in stem_to_indices.items():
        sources = {source_path(documents[index]) for index in group}
        if len(sources) < 2:
            continue
        kept_index = choose_filename_stem_representative(group, documents)
        kept_source = source_path(documents[kept_index])
        group_removed_indices = [
            index
            for index in group
            if source_path(documents[index]) != kept_source
        ]
        removed_sources = sorted({source_path(documents[index]) for index in group_removed_indices})
        removed_indices.update(group_removed_indices)
        filename_stem_reports.append(
            {
                "stem": stem,
                "kept": kept_source,
                "removed": removed_sources,
                "reason": "same_filename_stem_prefer_docx",
            }
        )

    kept_documents = [
        document
        for index, document in enumerate(documents)
        if index not in removed_indices
    ]
    return kept_documents, filename_stem_reports, removed_indices


def deduplicate_documents(documents: list[dict], threshold: float) -> tuple[list[dict], dict]:
    prepared_documents = prepare_documents(documents)
    duplicate_groups, pair_scores = detect_duplicate_groups(prepared_documents, threshold)

    removed_indices: set[int] = set()
    duplicate_group_reports: list[dict] = []
    for group in duplicate_groups:
        group_report, _kept_index, group_removed_indices = build_duplicate_group_report(
            group,
            prepared_documents,
            pair_scores,
        )
        duplicate_group_reports.append(group_report)
        removed_indices.update(group_removed_indices)

    kept_documents = [
        document
        for index, document in enumerate(documents)
        if index not in removed_indices
    ]

    stem_deduped_documents, filename_stem_reports, stem_removed_indices = deduplicate_by_filename_stem(kept_documents)
    total_removed_count = len(removed_indices) + len(stem_removed_indices)

    report = {
        "input_document_count": len(documents),
        "output_document_count": len(stem_deduped_documents),
        "removed_duplicate_count": total_removed_count,
        "threshold": threshold,
        "duplicate_groups": duplicate_group_reports,
        "filename_stem_duplicate_groups": filename_stem_reports,
        "examples": duplicate_group_reports[:10],
    }
    return stem_deduped_documents, report


def save_json(data: object, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def print_summary(report: dict, output_path: Path, report_path: Path) -> None:
    print("Document deduplication summary")
    print("==============================")
    print(f"Original document count: {report['input_document_count']}")
    print(f"Deduplicated document count: {report['output_document_count']}")
    print(f"Removed duplicate count: {report['removed_duplicate_count']}")
    print(f"Duplicate groups: {len(report['duplicate_groups'])}")
    print(f"Filename-stem duplicate groups: {len(report['filename_stem_duplicate_groups'])}")
    print(f"Threshold: {report['threshold']}")
    print(f"Deduplicated documents saved to: {output_path}")
    print(f"Dedup report path: {report_path}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    documents = load_documents(input_path)
    kept_documents, report = deduplicate_documents(documents, threshold=args.threshold)
    save_json(kept_documents, output_path)
    save_json(report, report_path)
    print_summary(report, output_path, report_path)


if __name__ == "__main__":
    main()
