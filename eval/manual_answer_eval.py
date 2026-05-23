from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation import generate_answer  # noqa: E402
from hybrid_retrieval import retrieve_hybrid  # noqa: E402


DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
DEFAULT_JSON_OUTPUT_PATH = PROJECT_ROOT / "eval" / "manual_answer_eval.json"
DEFAULT_MD_OUTPUT_PATH = PROJECT_ROOT / "eval" / "manual_answer_eval.md"
DEFAULT_K = 5
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_K = 30
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 2.0


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG answers for manual answer-quality inspection.")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Gold set JSONL path. Defaults to eval/gold_set.jsonl.",
    )
    parser.add_argument(
        "--question-numbers",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated 1-based question numbers to inspect.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="JSON output path. Defaults to eval/manual_answer_eval.json.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_OUTPUT_PATH,
        help="Markdown output path. Defaults to eval/manual_answer_eval.md.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Final retrieval depth. Defaults to 5.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Hybrid candidate depth. Defaults to the current recommended value, 50.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant. Defaults to the current recommended value, 30.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Dense retrieval weight. Defaults to the current recommended value, 0.5.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
        help="BM25 retrieval weight. Defaults to the current recommended value, 2.0.",
    )
    return parser.parse_args()


def parse_question_numbers(value: str) -> list[int]:
    numbers: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            number = int(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid question number: {stripped}") from exc
        if number <= 0:
            raise ValueError("Question numbers must be positive 1-based integers.")
        numbers.append(number)

    if not numbers:
        raise ValueError("At least one question number is required.")
    return list(dict.fromkeys(numbers))


def load_gold_set(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
            validate_gold_row(row, line_number)
            rows.append(row)
    return rows


def validate_gold_row(row: dict[str, Any], line_number: int) -> None:
    required_fields = {
        "question": str,
        "reference_answer": str,
        "must_cite_chunk_ids": list,
        "category": str,
    }
    for field, expected_type in required_fields.items():
        if field not in row:
            raise ValueError(f"Missing field '{field}' on gold set line {line_number}.")
        if not isinstance(row[field], expected_type):
            raise ValueError(f"Field '{field}' on gold set line {line_number} has the wrong type.")


def source_identifier(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "")
    chunk_id = chunk.get("chunk_id", "")
    if source and chunk_id:
        return f"{source} [{chunk_id}]"
    return str(source or chunk_id)


def selected_gold_rows(gold_rows: list[dict[str, Any]], question_numbers: list[int]) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    total = len(gold_rows)
    for question_number in question_numbers:
        if question_number > total:
            raise ValueError(f"Question number {question_number} is out of range. Gold set has {total} questions.")
        selected.append((question_number, gold_rows[question_number - 1]))
    return selected


def run_question(
    question_number: int,
    gold_row: dict[str, Any],
    k: int,
    candidate_k: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> dict[str, Any]:
    retrieved_chunks = retrieve_hybrid(
        gold_row["question"],
        k=k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    generated_answer = generate_answer(gold_row["question"], retrieved_chunks)

    return {
        "question_number": question_number,
        "question": gold_row["question"],
        "reference_answer": gold_row["reference_answer"],
        "expected_chunk_ids": gold_row["must_cite_chunk_ids"],
        "generated_answer": generated_answer,
        "sources": list(dict.fromkeys(source_identifier(chunk) for chunk in retrieved_chunks)),
        "retrieved_chunk_ids": [chunk.get("chunk_id", "") for chunk in retrieved_chunks],
        "manual_label": None,
        "notes": "",
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def markdown_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- `{item}`" for item in items)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Manual Answer Evaluation",
        "",
        "This file is for human inspection only. No automatic answer grading was applied.",
        "",
        "## Run Config",
        "",
        f"- Gold path: `{report['run_config']['gold_path']}`",
        f"- k: `{report['run_config']['k']}`",
        f"- candidate_k: `{report['run_config']['candidate_k']}`",
        f"- rrf_k: `{report['run_config']['rrf_k']}`",
        f"- dense_weight: `{report['run_config']['dense_weight']}`",
        f"- bm25_weight: `{report['run_config']['bm25_weight']}`",
        "",
        "## Results",
        "",
    ]

    for result in report["results"]:
        lines.extend(
            [
                f"### Question {result['question_number']}",
                "",
                f"Manual label: `{result['manual_label']}`",
                f"Notes: {result['notes']}",
                "",
                f"Question: {result['question']}",
                "",
                f"Reference answer: {result['reference_answer']}",
                "",
                f"Generated answer: {result['generated_answer']}",
                "",
                "Expected chunk IDs:",
                markdown_list(result["expected_chunk_ids"]),
                "",
                "Retrieved chunk IDs:",
                markdown_list(result["retrieved_chunk_ids"]),
                "",
                "Sources:",
                markdown_list(result["sources"]),
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gold_path = args.gold_path.resolve()
    question_numbers = parse_question_numbers(args.question_numbers)
    gold_rows = load_gold_set(gold_path)
    selected_rows = selected_gold_rows(gold_rows, question_numbers)

    results = [
        run_question(
            question_number=question_number,
            gold_row=gold_row,
            k=args.k,
            candidate_k=max(args.candidate_k, args.k),
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )
        for question_number, gold_row in selected_rows
    ]

    return {
        "run_config": {
            "gold_path": str(gold_path),
            "question_numbers": question_numbers,
            "retrieval_mode": "hybrid",
            "k": args.k,
            "candidate_k": max(args.candidate_k, args.k),
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
        },
        "results": results,
    }


def main() -> None:
    configure_stdout()
    args = parse_args()
    report = build_report(args)
    write_json(report, args.json_output)
    write_markdown(report, args.md_output)

    print("Manual answer evaluation complete")
    print("=================================")
    print(f"Questions evaluated: {len(report['results'])}")
    print(f"JSON saved to: {args.json_output}")
    print(f"Markdown saved to: {args.md_output}")


if __name__ == "__main__":
    main()
