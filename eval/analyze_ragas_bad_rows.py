"""
Analyze weak RAGAS rows for manual review.

Usage examples:
python eval/analyze_ragas_bad_rows.py --input eval/ragas_results_all_metrics_50_filled.csv --output eval/ragas_bad_rows_analysis.csv
python eval/analyze_ragas_bad_rows.py --input eval/ragas_results_all_metrics_50_filled.json --output eval/ragas_bad_rows_analysis.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "eval" / "ragas_results_all_metrics_50_filled.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "ragas_bad_rows_analysis.csv"
METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
QUESTION_ALIASES = ["question", "user_input", "query"]
ANSWER_ALIASES = ["answer", "response", "generated_answer"]
REFERENCE_ALIASES = ["reference", "ground_truth", "expected_answer"]
CONTEXTS_ALIASES = ["contexts", "retrieved_contexts"]
RETRIEVED_CONTEXTS_ALIASES = ["retrieved_contexts", "contexts"]
LIST_LIKE_COLUMNS = [
    "contexts",
    "retrieved_contexts",
    "expected_chunk_ids",
    "retrieved_chunk_ids",
    "sources",
]
MANUAL_COLUMNS = [
    "diagnosis",
    "suspected_issue",
    "manual_label",
    "recommended_fix",
]
SUSPECTED_ISSUE_OPTIONS = (
    "prompt_problem | answer_not_grounded | answer_not_direct | noisy_context | "
    "missing_context | bad_gold_answer | hebrew_eval_issue | data_format_issue"
)
CONTEXT_RECOVERY_NOTE = (
    "Full context text was not found in the input results file. Inspect run_ragas_eval.py "
    "or rerun evaluation with context persistence enabled."
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze weak RAGAS rows for manual review.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input evaluation file. Supports CSV, JSON, JSONL, and Parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for weak-row analysis.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return load_json_table(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def load_json_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return pd.DataFrame(payload["results"])
        raise ValueError("JSON input must be a list of rows or contain a top-level 'results' list.")
    raise ValueError("JSON input must contain row objects.")


def first_existing_column(columns: list[str], aliases: list[str]) -> str | None:
    existing = set(columns)
    for alias in aliases:
        if alias in existing:
            return alias
    return None


def ensure_text_series(frame: pd.DataFrame, aliases: list[str], default: str = "") -> tuple[pd.Series, str | None]:
    column = first_existing_column(frame.columns.tolist(), aliases)
    if column is None:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object"), None
    series = frame[column].copy()
    return series.map(lambda value: normalize_scalar(value, default=default)), column


def ensure_list_text_series(frame: pd.DataFrame, aliases: list[str]) -> tuple[pd.Series, str | None]:
    column = first_existing_column(frame.columns.tolist(), aliases)
    if column is None:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object"), None
    return frame[column].map(normalize_list_like), column


def normalize_scalar(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_list_like(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped
    return str(value)


def parse_metric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column_name], errors="coerce").astype("Float64")


def print_metric_averages(frame: pd.DataFrame) -> None:
    print("Metric averages")
    print("===============")
    for metric_name in METRIC_COLUMNS:
        if metric_name not in frame.columns:
            print(f"{metric_name}: unavailable (column missing)")
            continue
        numeric = parse_metric_series(frame, metric_name).dropna()
        if numeric.empty:
            print(f"{metric_name}: unavailable (no numeric values)")
            continue
        print(f"{metric_name}: {float(numeric.mean()):.4f}")
    print()


def print_missing_expected_columns(frame: pd.DataFrame) -> None:
    expected_groups = {
        "question_number": ["question_number"],
        "category": ["category"],
        "question": QUESTION_ALIASES,
        "answer": ANSWER_ALIASES,
        "reference": REFERENCE_ALIASES,
        "ground_truth": ["ground_truth", "reference", "expected_answer"],
        "contexts": CONTEXTS_ALIASES,
        "retrieved_contexts": RETRIEVED_CONTEXTS_ALIASES,
        "retrieved_context_count": ["retrieved_context_count"],
        "expected_chunk_ids": ["expected_chunk_ids"],
        "retrieved_chunk_ids": ["retrieved_chunk_ids"],
        "sources": ["sources"],
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy"],
        "context_precision": ["context_precision"],
        "context_recall": ["context_recall"],
    }
    missing = [
        canonical_name
        for canonical_name, aliases in expected_groups.items()
        if first_existing_column(frame.columns.tolist(), aliases) is None
    ]
    if not missing:
        return
    print("Warning: Missing expected columns in input:")
    for column in missing:
        print(f"- {column}")
    print()


def weak_mask(frame: pd.DataFrame) -> pd.Series:
    faithfulness = parse_metric_series(frame, "faithfulness")
    answer_relevancy = parse_metric_series(frame, "answer_relevancy")
    return ((faithfulness < 0.6).fillna(False)) | ((answer_relevancy < 0.6).fillna(False))


def print_summary(frame: pd.DataFrame, weak_rows: pd.DataFrame) -> None:
    total_rows = len(frame)
    weak_count = len(weak_rows)
    weak_percentage = (weak_count / total_rows * 100.0) if total_rows else 0.0

    print("Row summary")
    print("===========")
    print(f"Total rows: {total_rows}")
    print(f"Weak rows count: {weak_count}")
    print(f"Weak rows percentage: {weak_percentage:.2f}%")
    print()


def weakest_rows(frame: pd.DataFrame, metric_name: str, limit: int = 5) -> pd.DataFrame:
    if metric_name not in frame.columns:
        return pd.DataFrame()
    ranked = frame.copy()
    ranked[metric_name] = parse_metric_series(frame, metric_name)
    ranked = ranked.dropna(subset=[metric_name]).sort_values(by=metric_name, ascending=True)
    return ranked.head(limit)


def print_weakest_questions(frame: pd.DataFrame, metric_name: str) -> None:
    print(f"5 weakest questions by {metric_name}")
    print("=" * (24 + len(metric_name)))
    weakest = weakest_rows(frame, metric_name)
    if weakest.empty:
        print("No numeric rows available.")
        print()
        return

    for _, row in weakest.iterrows():
        question_number = normalize_scalar(row.get("question_number", ""))
        question = normalize_scalar(row.get("question", ""))
        score = row.get(metric_name)
        print(f"- Q{question_number} | {metric_name}={float(score):.4f} | {question}")
    print()


def build_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    analysis = pd.DataFrame(index=frame.index)

    analysis["question_number"] = frame["question_number"] if "question_number" in frame.columns else pd.Series(
        [""] * len(frame), index=frame.index, dtype="object"
    )
    analysis["category"] = frame["category"] if "category" in frame.columns else pd.Series(
        [""] * len(frame), index=frame.index, dtype="object"
    )

    analysis["question"], _ = ensure_text_series(frame, QUESTION_ALIASES)
    analysis["answer"], _ = ensure_text_series(frame, ANSWER_ALIASES)
    analysis["reference"], _ = ensure_text_series(frame, REFERENCE_ALIASES)
    analysis["ground_truth"], _ = ensure_text_series(frame, ["ground_truth", "reference", "expected_answer"])

    contexts_series, contexts_column = ensure_list_text_series(frame, CONTEXTS_ALIASES)
    retrieved_contexts_series, retrieved_contexts_column = ensure_list_text_series(frame, RETRIEVED_CONTEXTS_ALIASES)

    analysis["contexts"] = contexts_series
    analysis["retrieved_contexts"] = retrieved_contexts_series
    analysis["retrieved_context_count"] = (
        frame["retrieved_context_count"]
        if "retrieved_context_count" in frame.columns
        else pd.Series([""] * len(frame), index=frame.index, dtype="object")
    )
    analysis["expected_chunk_ids"], _ = ensure_list_text_series(frame, ["expected_chunk_ids"])
    analysis["retrieved_chunk_ids"], _ = ensure_list_text_series(frame, ["retrieved_chunk_ids"])
    analysis["sources"], _ = ensure_list_text_series(frame, ["sources"])

    context_columns_present = contexts_column is not None or retrieved_contexts_column is not None
    if not context_columns_present:
        print("Warning: Full context text columns were not found in the input file.")
        analysis["context_text_available"] = False
        analysis["context_recovery_note"] = CONTEXT_RECOVERY_NOTE
    else:
        analysis["context_text_available"] = True
        analysis["context_recovery_note"] = ""

    for metric_name in METRIC_COLUMNS:
        analysis[metric_name] = parse_metric_series(frame, metric_name)

    for column_name in MANUAL_COLUMNS:
        analysis[column_name] = ""
    analysis["suspected_issue_options"] = SUSPECTED_ISSUE_OPTIONS

    ordered_columns = [
        "question_number",
        "category",
        "question",
        "answer",
        "reference",
        "ground_truth",
        "contexts",
        "retrieved_contexts",
        "retrieved_context_count",
        "expected_chunk_ids",
        "retrieved_chunk_ids",
        "sources",
        "context_text_available",
        "context_recovery_note",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "diagnosis",
        "suspected_issue",
        "suspected_issue_options",
        "manual_label",
        "recommended_fix",
    ]
    return analysis[ordered_columns]


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    configure_stdout()
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    frame = load_table(input_path)
    if frame.empty:
        raise ValueError(f"Input file is empty: {input_path}")

    analysis = build_analysis_frame(frame)
    print_missing_expected_columns(frame)
    print_metric_averages(analysis)

    mask = weak_mask(analysis)
    weak_rows = analysis.loc[mask].copy()

    print_summary(analysis, weak_rows)
    print_weakest_questions(analysis, "faithfulness")
    print_weakest_questions(analysis, "answer_relevancy")

    save_csv(weak_rows, output_path)
    print(f"Saved weak-row analysis to: {output_path}")


if __name__ == "__main__":
    main()
