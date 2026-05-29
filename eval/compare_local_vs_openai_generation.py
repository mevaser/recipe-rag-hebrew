from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_utils import print_openai_api_key_status  # noqa: E402
from eval_answer_backends import (  # noqa: E402
    DEFAULT_OPENAI_ANSWER_MODEL,
    generate_strict_answer,
)


DEFAULT_INPUT_PATH = PROJECT_ROOT / "eval" / "context_ordering_experiment_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "local_vs_openai_generation_comparison.csv"
DEFAULT_QUESTION_NUMBERS = ["29", "31", "41", "50", "24", "25", "27", "36"]
DEFAULT_CONTEXT_VARIANTS = [
    "current_top_5",
    "top_3_only",
    "oracle_expected_chunk_if_available",
]
OUTPUT_COLUMNS = [
    "question_number",
    "context_variant",
    "reference",
    "human_main_issue",
    "local_answer",
    "openai_answer",
    "local_model",
    "openai_model",
    "openai_better_than_local",
    "manual_notes",
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local and OpenAI answer generation on frozen context variants."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input CSV path. Defaults to eval/context_ordering_experiment_dataset.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path. Defaults to eval/local_vs_openai_generation_comparison.csv.",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="qwen2.5:7b-instruct",
        help="Local answer model name. Defaults to qwen2.5:7b-instruct.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=DEFAULT_OPENAI_ANSWER_MODEL,
        help=f"OpenAI answer model name. Defaults to {DEFAULT_OPENAI_ANSWER_MODEL}.",
    )
    parser.add_argument(
        "--question-numbers",
        type=str,
        default=",".join(DEFAULT_QUESTION_NUMBERS),
        help="Comma-separated question numbers to compare.",
    )
    parser.add_argument(
        "--context-variants",
        type=str,
        default=",".join(DEFAULT_CONTEXT_VARIANTS),
        help="Comma-separated context variants to compare.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after filtering.",
    )
    parser.add_argument(
        "--generation-context-k",
        type=int,
        default=None,
        help="Optional cap on how many frozen contexts are passed into each backend.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output CSV if it already exists.",
    )
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
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


def parse_context_list(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, list):
        return [normalize_text(item) for item in parsed if normalize_text(item)]
    normalized = normalize_text(parsed)
    return [normalized] if normalized else []


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def select_rows(
    frame: pd.DataFrame,
    question_numbers: list[str],
    context_variants: list[str],
    limit: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    working = frame.copy()
    working["question_number_normalized"] = working["question_number"].map(normalize_question_number)
    working["context_variant_normalized"] = working["context_variant"].map(normalize_text)
    requested_numbers = set(question_numbers)
    selected = working[
        working["question_number_normalized"].isin(requested_numbers)
        & working["context_variant_normalized"].isin(set(context_variants))
    ].copy()
    selected = selected[selected["variant_contexts"].map(parse_context_list).map(bool)]
    if limit is not None:
        selected = selected.head(limit).copy()

    found_numbers = set(selected["question_number_normalized"].tolist())
    missing_numbers = [question_number for question_number in question_numbers if question_number not in found_numbers]
    return selected, missing_numbers


def build_output_rows(
    frame: pd.DataFrame,
    local_model: str,
    openai_model: str,
    generation_context_k: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        contexts = parse_context_list(row.get("variant_contexts", ""))
        question = normalize_text(row.get("question", ""))
        rows.append(
            {
                "question_number": row.get("question_number", ""),
                "context_variant": row.get("context_variant", ""),
                "reference": row.get("reference", ""),
                "human_main_issue": row.get("human_main_issue", ""),
                "local_answer": generate_strict_answer(
                    question=question,
                    contexts=contexts,
                    answer_backend="local",
                    answer_model=local_model,
                    generation_context_k=generation_context_k,
                ),
                "openai_answer": generate_strict_answer(
                    question=question,
                    contexts=contexts,
                    answer_backend="openai",
                    answer_model=openai_model,
                    generation_context_k=generation_context_k,
                ),
                "local_model": local_model,
                "openai_model": openai_model,
                "openai_better_than_local": "",
                "manual_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_output(frame: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    configure_stdout()
    print_openai_api_key_status(__file__)
    args = parse_args()

    question_numbers = [normalize_question_number(value) for value in parse_csv_list(args.question_numbers)]
    context_variants = parse_csv_list(args.context_variants)
    frame = load_frame(args.input)
    selected, missing_numbers = select_rows(
        frame,
        question_numbers=question_numbers,
        context_variants=context_variants,
        limit=args.limit,
    )
    output = build_output_rows(
        selected,
        local_model=args.local_model,
        openai_model=args.openai_model,
        generation_context_k=args.generation_context_k,
    )
    write_output(output, args.output, overwrite=args.overwrite)

    print(f"Selected rows: {len(output)}")
    print(
        "Generation context k: "
        + ("all available contexts" if args.generation_context_k is None else str(args.generation_context_k))
    )
    print("Selected question_numbers: " + ", ".join(output["question_number"].map(normalize_question_number).tolist()))
    print("Selected context variants: " + ", ".join(context_variants))
    if missing_numbers:
        print("Requested question_numbers not found in dataset: " + ", ".join(missing_numbers))
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
