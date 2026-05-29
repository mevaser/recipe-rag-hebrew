from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bm25_retrieval import retrieve_bm25  # noqa: E402
from hybrid_retrieval import retrieve_hybrid  # noqa: E402
from retrieval import retrieve as retrieve_dense  # noqa: E402


DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
DEFAULT_INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"
DEFAULT_DOCUMENTS_DEDUP_PATH = PROJECT_ROOT / "data" / "processed" / "documents_dedup.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "eval" / "missing_context_investigation.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "eval" / "missing_context_investigation_findings.md"
TARGET_QUESTION_NUMBERS = ["4", "5", "10", "35", "46"]
TOKEN_PATTERN = re.compile(r"[\w\u0590-\u05ff]+", flags=re.UNICODE)


@dataclass
class InvestigationRow:
    question_number: str
    question: str
    reference: str
    expected_source: str
    expected_chunk_found_in_processed_chunks: str
    expected_chunk_id: str
    expected_chunk_source: str
    expected_chunk_rank_dense: str
    expected_chunk_rank_bm25: str
    expected_chunk_rank_hybrid: str
    best_candidate_chunk_id: str
    best_candidate_source: str
    best_candidate_method: str
    best_candidate_rank: str
    best_candidate_text_preview: str
    suspected_reason: str
    recommended_fix: str
    manual_notes: str


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Investigate missing-context retrieval failures on a small fixed question set."
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Gold set JSONL path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path. Defaults to eval/missing_context_investigation.csv.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Output Markdown path. Defaults to eval/missing_context_investigation_findings.md.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_question_number(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def tokenize(text: str) -> set[str]:
    tokens = {token.casefold() for token in TOKEN_PATTERN.findall(normalize_text(text))}
    return {token for token in tokens if len(token) > 1}


def overlap_count(first: str, second: str) -> int:
    return len(tokenize(first) & tokenize(second))


def preview_text(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", normalize_text(text))
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def expected_source_from_chunk_id(chunk_id: str) -> str:
    if "_chunk_" in chunk_id:
        return chunk_id.rsplit("_chunk_", 1)[0]
    return chunk_id


def build_chunk_lookup(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_text(chunk.get("chunk_id", "")): chunk for chunk in chunks}


def build_document_lookup(documents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        doc_id = normalize_text(document.get("doc_id", ""))
        lookup.setdefault(doc_id, []).append(document)
    return lookup


def chunk_source_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    return " ".join(
        [
            normalize_text(chunk.get("doc_id", "")),
            normalize_text(chunk.get("chunk_id", "")),
            normalize_text(metadata.get("source", "")),
            normalize_text(metadata.get("relative_path", "")),
            normalize_text(metadata.get("category", "")),
        ]
    ).strip()


def best_candidate_from_chunks(
    question: str,
    reference: str,
    expected_source: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    expected_source_terms = tokenize(expected_source)
    expected_source_tail = Path(expected_source).stem if expected_source else ""
    best_score = -1.0
    best_chunk: dict[str, Any] | None = None
    for chunk in chunks:
        source_text = chunk_source_text(chunk)
        source_terms = tokenize(source_text)
        score = 0.0
        if expected_source and expected_source in normalize_text(chunk.get("chunk_id", "")):
            score += 100.0
        if expected_source and expected_source in source_text:
            score += 50.0
        score += 10.0 * len(expected_source_terms & source_terms)
        score += 3.0 * overlap_count(question, chunk.get("text", ""))
        score += 2.0 * overlap_count(reference, chunk.get("text", ""))
        if expected_source_tail and expected_source_tail.casefold() in source_text.casefold():
            score += 10.0
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def rank_of_chunk(results: list[dict[str, Any]], chunk_id: str) -> int | None:
    for index, result in enumerate(results, start=1):
        if normalize_text(result.get("chunk_id", "")) == chunk_id:
            return index
    return None


def best_candidate_rank(
    candidate_chunk_id: str,
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> tuple[str, str]:
    candidate_ranks = [
        ("dense", rank_of_chunk(dense_results, candidate_chunk_id)),
        ("bm25", rank_of_chunk(bm25_results, candidate_chunk_id)),
        ("hybrid", rank_of_chunk(hybrid_results, candidate_chunk_id)),
    ]
    available = [(method, rank) for method, rank in candidate_ranks if rank is not None]
    if not available:
        return "", ""
    method, rank = min(available, key=lambda item: item[1])
    return method, str(rank)


def retrieval_results_by_method(question: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "dense": retrieve_dense(question, k=100),
        "bm25": retrieve_bm25(question, k=100),
        "hybrid": retrieve_hybrid(question, k=100, candidate_k=100),
    }


def document_matches_reference(
    expected_source: str,
    reference: str,
    documents: list[dict[str, Any]],
) -> bool:
    expected_terms = tokenize(expected_source)
    reference_terms = tokenize(reference)
    for document in documents:
        metadata = document.get("metadata", {})
        source_text = " ".join(
            [
                normalize_text(document.get("doc_id", "")),
                normalize_text(metadata.get("source", "")),
                normalize_text(metadata.get("relative_path", "")),
            ]
        )
        if expected_terms and not (expected_terms & tokenize(source_text)):
            continue
        if overlap_count(reference, document.get("text", "")) >= max(2, min(5, len(reference_terms))):
            return True
    return False


def infer_suspected_reason(
    expected_chunk_found: bool,
    expected_chunk_rank_dense: int | None,
    expected_chunk_rank_bm25: int | None,
    expected_chunk_rank_hybrid: int | None,
    expected_source: str,
    best_candidate_chunk: dict[str, Any] | None,
    raw_has_reference: bool,
    question: str,
    reference: str,
) -> tuple[str, str]:
    if not expected_chunk_found:
        if raw_has_reference:
            if best_candidate_chunk is None:
                return "answer_exists_in_raw_but_not_processed", "inspect_raw_document_extraction"
            return "source_metadata_mismatch", "improve_metadata_extraction"
        return "expected_chunk_not_in_processed_chunks", "inspect_raw_document_extraction"

    if expected_chunk_rank_dense is not None and expected_chunk_rank_dense > 5:
        if expected_chunk_rank_hybrid is not None and expected_chunk_rank_hybrid <= 5:
            return "expected_chunk_exists_but_ranked_low", "add_source_title_boosting"
        if expected_chunk_rank_bm25 is not None and expected_chunk_rank_bm25 <= 5:
            return "expected_chunk_exists_but_ranked_low", "add_source_title_boosting"

    if expected_chunk_rank_hybrid is not None and expected_chunk_rank_hybrid > 20:
        return "expected_chunk_exists_but_ranked_low", "add_source_title_boosting"

    if best_candidate_chunk is not None:
        source_text = chunk_source_text(best_candidate_chunk)
        if tokenize(expected_source) and not (tokenize(expected_source) & tokenize(source_text)):
            return "source_metadata_mismatch", "improve_metadata_extraction"
        if overlap_count(question, best_candidate_chunk.get("text", "")) <= 1 and overlap_count(
            reference, best_candidate_chunk.get("text", "")
        ) <= 1:
            return "query_terms_do_not_match_chunk_terms", "add_query_expansion"

    if len(tokenize(question)) <= 3:
        return "ambiguous_question", "no_action_until_manual_review"
    return "unknown", "no_action_until_manual_review"


def build_row(
    question_number: str,
    gold_row: dict[str, Any],
    all_chunks: list[dict[str, Any]],
    indexed_chunks: list[dict[str, Any]],
    all_documents: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
) -> InvestigationRow:
    question = normalize_text(gold_row.get("question", ""))
    reference = normalize_text(gold_row.get("reference_answer", ""))
    expected_chunk_id = normalize_text((gold_row.get("must_cite_chunk_ids") or [""])[0])
    expected_source = expected_source_from_chunk_id(expected_chunk_id)

    expected_chunk = chunk_lookup.get(expected_chunk_id)
    expected_chunk_found = expected_chunk is not None
    expected_chunk_source = ""
    if expected_chunk is not None:
        metadata = expected_chunk.get("metadata", {})
        expected_chunk_source = normalize_text(metadata.get("source", "")) or normalize_text(
            metadata.get("relative_path", "")
        )

    retrieval_results = retrieval_results_by_method(question)
    dense_results = retrieval_results["dense"]
    bm25_results = retrieval_results["bm25"]
    hybrid_results = retrieval_results["hybrid"]

    expected_rank_dense = rank_of_chunk(dense_results, expected_chunk_id) if expected_chunk_found else None
    expected_rank_bm25 = rank_of_chunk(bm25_results, expected_chunk_id) if expected_chunk_found else None
    expected_rank_hybrid = rank_of_chunk(hybrid_results, expected_chunk_id) if expected_chunk_found else None

    best_candidate_chunk = best_candidate_from_chunks(question, reference, expected_source, indexed_chunks)
    best_candidate_chunk_id = normalize_text(best_candidate_chunk.get("chunk_id", "")) if best_candidate_chunk else ""
    best_candidate_source = ""
    if best_candidate_chunk:
        best_candidate_source = normalize_text(best_candidate_chunk.get("metadata", {}).get("source", "")) or normalize_text(
            best_candidate_chunk.get("metadata", {}).get("relative_path", "")
        )
    best_candidate_method, best_candidate_rank_value = best_candidate_rank(
        best_candidate_chunk_id,
        dense_results,
        bm25_results,
        hybrid_results,
    )

    raw_has_reference = document_matches_reference(expected_source, reference, all_documents)
    suspected_reason, recommended_fix = infer_suspected_reason(
        expected_chunk_found=expected_chunk_found,
        expected_chunk_rank_dense=expected_rank_dense,
        expected_chunk_rank_bm25=expected_rank_bm25,
        expected_chunk_rank_hybrid=expected_rank_hybrid,
        expected_source=expected_source,
        best_candidate_chunk=best_candidate_chunk,
        raw_has_reference=raw_has_reference,
        question=question,
        reference=reference,
    )

    return InvestigationRow(
        question_number=question_number,
        question=question,
        reference=reference,
        expected_source=expected_source,
        expected_chunk_found_in_processed_chunks="yes" if expected_chunk_found else "no",
        expected_chunk_id=expected_chunk_id,
        expected_chunk_source=expected_chunk_source,
        expected_chunk_rank_dense="" if expected_rank_dense is None else str(expected_rank_dense),
        expected_chunk_rank_bm25="" if expected_rank_bm25 is None else str(expected_rank_bm25),
        expected_chunk_rank_hybrid="" if expected_rank_hybrid is None else str(expected_rank_hybrid),
        best_candidate_chunk_id=best_candidate_chunk_id,
        best_candidate_source=best_candidate_source,
        best_candidate_method=best_candidate_method,
        best_candidate_rank=best_candidate_rank_value,
        best_candidate_text_preview="" if best_candidate_chunk is None else preview_text(best_candidate_chunk.get("text", "")),
        suspected_reason=suspected_reason,
        recommended_fix=recommended_fix,
        manual_notes="",
    )


def write_csv(rows: list[InvestigationRow], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.__dict__ for row in rows]).to_csv(output_path, index=False, encoding="utf-8-sig")


def write_markdown(rows: list[InvestigationRow], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")
    counts = Counter(row.suspected_reason for row in rows)
    lines = [
        "# Missing Context Investigation",
        "",
        "## Goal",
        "Investigate rows where retrieval failed to provide the expected context.",
        "",
        "## Questions Investigated",
        "- Q4",
        "- Q5",
        "- Q10",
        "- Q35",
        "- Q46",
        "",
        "## Per-Question Findings",
    ]
    for row in rows:
        lines.extend(
            [
                f"### Q{row.question_number}",
                f"- question: {row.question}",
                f"- expected/reference answer: {row.reference}",
                f"- expected chunk exists in processed chunks: {row.expected_chunk_found_in_processed_chunks}",
                f"- appears in dense top results: {'yes' if row.expected_chunk_rank_dense else 'no'}",
                f"- appears in BM25 top results: {'yes' if row.expected_chunk_rank_bm25 else 'no'}",
                f"- appears in hybrid top results: {'yes' if row.expected_chunk_rank_hybrid else 'no'}",
                f"- expected chunk rank in dense: {row.expected_chunk_rank_dense or 'not found'}",
                f"- expected chunk rank in BM25: {row.expected_chunk_rank_bm25 or 'not found'}",
                f"- expected chunk rank in hybrid: {row.expected_chunk_rank_hybrid or 'not found'}",
                f"- suspected reason: {row.suspected_reason}",
                f"- recommended fix: {row.recommended_fix}",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            *[f"- {reason}: {count}" for reason, count in sorted(counts.items())],
            "",
            "## Next Retrieval Fix Candidates",
            "1. source/title/recipe-name boosting",
            "2. query expansion",
            "3. metadata extraction improvements",
            "4. Hebrew/RTL text normalization",
            "5. raw document extraction inspection",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_stdout()
    args = parse_args()

    gold_rows = load_gold_rows(args.gold_path)
    indexed_chunks = load_json(DEFAULT_INDEXED_CHUNKS_PATH)
    all_chunks = load_json(DEFAULT_CHUNKS_PATH)
    all_documents = load_json(DEFAULT_DOCUMENTS_PATH) + load_json(DEFAULT_DOCUMENTS_DEDUP_PATH)
    chunk_lookup = build_chunk_lookup(all_chunks + indexed_chunks)

    rows: list[InvestigationRow] = []
    for question_number in TARGET_QUESTION_NUMBERS:
        gold_index = int(question_number) - 1
        gold_row = gold_rows[gold_index]
        rows.append(
            build_row(
                question_number=question_number,
                gold_row=gold_row,
                all_chunks=all_chunks,
                indexed_chunks=indexed_chunks,
                all_documents=all_documents,
                chunk_lookup=chunk_lookup,
            )
        )

    write_csv(rows, args.output_csv, overwrite=args.overwrite)
    write_markdown(rows, args.output_md, overwrite=args.overwrite)

    counts = Counter(row.suspected_reason for row in rows)
    print(f"Questions analyzed: {len(rows)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output Markdown: {args.output_md}")
    print(
        "Suspected reasons: "
        + ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
    )


if __name__ == "__main__":
    main()
