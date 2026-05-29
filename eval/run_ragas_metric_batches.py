from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_ragas_eval import (
    DEFAULT_EVAL_PROMPT_VERSION,
    DEFAULT_GOLD_PATH,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_METRICS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OUTPUT_DIR,
    configure_stdout,
    flatten_row,
    load_jsonl,
    normalized_provider,
    parse_metric_names,
    resolve_answer_model_name,
    sanitize_model_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_RAGAS_EVAL_SCRIPT = PROJECT_ROOT / "eval" / "run_ragas_eval.py"
DEFAULT_TOTAL = 50
DEFAULT_BATCH_SIZE = 5
DEFAULT_OUTPUT_DIR_METRICS = DEFAULT_OUTPUT_DIR / "ragas_metric_batches"
BASE_COLUMNS = [
    "question_number",
    "category",
    "question",
    "answer",
    "reference",
    "contexts",
    "retrieved_contexts",
    "retrieved_context_count",
    "expected_chunk_ids",
    "retrieved_chunk_ids",
    "sources",
]


def parse_args() -> argparse.Namespace:
    env_provider = normalized_provider(os.getenv("RAGAS_LLM_PROVIDER", "ollama"))
    parser = argparse.ArgumentParser(description="Run metric-by-metric batched RAGAS evaluation.")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Gold set JSONL path. Defaults to eval/gold_set.jsonl.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=DEFAULT_TOTAL,
        help="Maximum number of examples to evaluate. Defaults to 50.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Examples per batch. Defaults to 5.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=DEFAULT_METRICS,
        help="Comma-separated metric names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for per-metric batch outputs. Defaults to a prompt-version-safe directory.",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="Combined CSV output path. Defaults to a prompt-version-safe filename.",
    )
    parser.add_argument(
        "--combined-json",
        type=Path,
        default=None,
        help="Combined JSON output path. Defaults to a prompt-version-safe filename.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=env_provider,
        help="Evaluator provider. Defaults to RAGAS_LLM_PROVIDER or ollama.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("RAGAS_LLM_MODEL", DEFAULT_OLLAMA_MODEL),
        help="Evaluator model. Defaults to RAGAS_LLM_MODEL or qwen2.5:7b-instruct.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL),
        help="Embedding model. Defaults to RAGAS_EMBEDDING_MODEL or intfloat/multilingual-e5-small.",
    )
    parser.add_argument(
        "--continue-on-error",
        type=parse_bool,
        default=True,
        help="Continue after failed batches. Defaults to true.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=os.getenv("PROMPT_VERSION", DEFAULT_EVAL_PROMPT_VERSION),
        help="Answer-generation prompt version. Defaults to PROMPT_VERSION or strict_short_no_sources.",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default=os.getenv("ANSWER_MODEL", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)),
        help="Answer-generation model name. Defaults to ANSWER_MODEL, OLLAMA_MODEL, or qwen2.5:7b-instruct.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("--continue-on-error must be true or false.")


def build_batch_ranges(total_examples: int, batch_size: int) -> list[tuple[int, int]]:
    if total_examples <= 0:
        raise ValueError("--total must be greater than 0.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    ranges: list[tuple[int, int]] = []
    for start in range(0, total_examples, batch_size):
        limit = min(batch_size, total_examples - start)
        ranges.append((start, limit))
    return ranges


def metric_batch_stem(metric_name: str, start: int, limit: int) -> str:
    end = start + limit - 1
    return f"ragas_{metric_name}_{start:03d}_{end:03d}"


def metric_dir(root_output_dir: Path, metric_name: str) -> Path:
    return root_output_dir / metric_name


def metric_batch_paths(root_output_dir: Path, metric_name: str, start: int, limit: int) -> tuple[Path, Path]:
    stem = metric_batch_stem(metric_name, start, limit)
    directory = metric_dir(root_output_dir, metric_name)
    return directory / f"{stem}.csv", directory / f"{stem}.json"


def metric_merged_paths(root_output_dir: Path, metric_name: str) -> tuple[Path, Path]:
    directory = metric_dir(root_output_dir, metric_name)
    return directory / f"ragas_{metric_name}_merged.csv", directory / f"ragas_{metric_name}_merged.json"


def default_metric_output_dir(prompt_version: str, answer_model: str) -> Path:
    normalized = prompt_version.strip().lower()
    if normalized == "baseline":
        return DEFAULT_OUTPUT_DIR_METRICS
    return DEFAULT_OUTPUT_DIR / f"ragas_metric_batches_{normalized}_{sanitize_model_name(answer_model)}"


def default_combined_csv_path(prompt_version: str, answer_model: str) -> Path:
    normalized = prompt_version.strip().lower()
    if normalized == "baseline":
        return DEFAULT_OUTPUT_DIR / "ragas_results_all_metrics_50.csv"
    return DEFAULT_OUTPUT_DIR / f"ragas_results_all_metrics_50_{normalized}_{sanitize_model_name(answer_model)}.csv"


def default_combined_json_path(prompt_version: str, answer_model: str) -> Path:
    normalized = prompt_version.strip().lower()
    if normalized == "baseline":
        return DEFAULT_OUTPUT_DIR / "ragas_results_all_metrics_50.json"
    return DEFAULT_OUTPUT_DIR / f"ragas_results_all_metrics_50_{normalized}_{sanitize_model_name(answer_model)}.json"


def run_metric_batch(
    args: argparse.Namespace,
    metric_name: str,
    start: int,
    limit: int,
    csv_path: Path,
    json_path: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(RUN_RAGAS_EVAL_SCRIPT),
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--metrics",
        metric_name,
        "--output-dir",
        str(csv_path.parent),
        "--output-csv",
        str(csv_path),
        "--output-json",
        str(json_path),
        "--gold-path",
        str(args.gold_path.resolve()),
        "--llm-provider",
        args.llm_provider,
        "--llm-model",
        args.llm_model,
        "--embedding-model",
        args.embedding_model,
        "--prompt-version",
        args.prompt_version,
        "--answer-model",
        args.answer_model,
    ]
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def write_error_batch_json(
    json_path: Path,
    metric_name: str,
    start: int,
    limit: int,
    process: subprocess.CompletedProcess[str] | None,
    error_message: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": metric_name,
        "start": start,
        "limit": limit,
        "end": start + limit - 1,
        "status": "failed",
        "error": error_message,
        "returncode": process.returncode if process is not None else None,
        "stdout": process.stdout if process is not None else "",
        "stderr": process.stderr if process is not None else "",
        "results": [],
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def write_error_batch_csv(csv_path: Path, metric_name: str, start: int, limit: int, error_message: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "question_number": "",
        "category": "",
        "question": "",
        "answer": "",
        "reference": "",
        "contexts": [],
        "retrieved_contexts": [],
        "retrieved_context_count": 0,
        "expected_chunk_ids": [],
        "retrieved_chunk_ids": [],
        "sources": [],
        metric_name: None,
        "batch_status": "failed",
        "batch_start": start,
        "batch_limit": limit,
        "batch_end": start + limit - 1,
        "error": error_message,
    }
    flattened = flatten_row(row)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(flattened.keys()))
        writer.writeheader()
        writer.writerow(flattened)


def load_batch_results(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing batch JSON output: {json_path}")

    with json_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Batch JSON output does not contain a results list: {json_path}")
    return [dict(row) for row in results]


def save_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened_rows = [flatten_row(row) for row in rows]
    fieldnames = list(flattened_rows[0].keys()) if flattened_rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_rows)


def save_rows_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def metric_average(rows: list[dict[str, Any]], metric_name: str) -> float | None:
    values = [
        float(row[metric_name])
        for row in rows
        if isinstance(row.get(metric_name), (int, float)) and not math.isnan(float(row[metric_name]))
    ]
    if not values:
        print(
            f"Warning: Metric {metric_name} returned only nan values, likely due to local evaluator JSON-format issues.",
            file=sys.stderr,
        )
        return None
    return sum(values) / len(values)


def initialize_combined_rows(gold_rows: list[dict[str, Any]], total_examples: int, metric_names: list[str]) -> dict[int, dict[str, Any]]:
    combined: dict[int, dict[str, Any]] = {}
    for question_number, example in enumerate(gold_rows[:total_examples], start=1):
        row: dict[str, Any] = {
            "question_number": question_number,
            "category": example.get("category", ""),
            "question": example.get("question", ""),
            "answer": "",
            "reference": example.get("reference_answer", ""),
            "contexts": [],
            "retrieved_contexts": [],
            "retrieved_context_count": 0,
            "expected_chunk_ids": example.get("must_cite_chunk_ids", []),
            "retrieved_chunk_ids": [],
            "sources": [],
        }
        for metric_name in metric_names:
            row[metric_name] = None
        combined[question_number] = row
    return combined


def merge_metric_rows(
    combined_rows: dict[int, dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    metric_name: str,
) -> None:
    for row in metric_rows:
        question_number = row.get("question_number")
        if not isinstance(question_number, int):
            continue
        if question_number not in combined_rows:
            print(
                f"Warning: Skipping out-of-range question_number {question_number} for metric {metric_name}.",
                file=sys.stderr,
            )
            continue

        target = combined_rows[question_number]
        for key in BASE_COLUMNS:
            value = row.get(key)
            if key == "retrieved_context_count":
                if isinstance(value, int) and value > 0:
                    target[key] = value
            elif key in {"contexts", "retrieved_contexts", "expected_chunk_ids", "retrieved_chunk_ids", "sources"}:
                if isinstance(value, list) and value:
                    target[key] = value
            elif isinstance(value, str) and value.strip():
                target[key] = value
        target[metric_name] = row.get(metric_name)


def main() -> None:
    configure_stdout()
    args = parse_args()
    resolved_answer_model = resolve_answer_model_name(args.answer_model)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else default_metric_output_dir(args.prompt_version, resolved_answer_model).resolve()
    )
    combined_csv_path = (
        args.combined_csv.resolve()
        if args.combined_csv
        else default_combined_csv_path(args.prompt_version, resolved_answer_model).resolve()
    )
    combined_json_path = (
        args.combined_json.resolve()
        if args.combined_json
        else default_combined_json_path(args.prompt_version, resolved_answer_model).resolve()
    )
    gold_rows = load_jsonl(args.gold_path.resolve())
    total_examples = min(args.total, len(gold_rows))
    metric_names = parse_metric_names(args.metrics)
    batch_ranges = build_batch_ranges(total_examples, args.batch_size)

    combined_rows = initialize_combined_rows(gold_rows, total_examples, metric_names)
    metric_summaries: list[dict[str, Any]] = []
    total_batches_attempted = 0
    successful_batches = 0
    failed_batches = 0

    for metric_name in metric_names:
        metric_rows: list[dict[str, Any]] = []
        metric_batch_records: list[dict[str, Any]] = []

        for batch_index, (start, limit) in enumerate(batch_ranges, start=1):
            total_batches_attempted += 1
            csv_path, json_path = metric_batch_paths(output_dir, metric_name, start, limit)
            print(f"Metric {metric_name} batch {batch_index}/{len(batch_ranges)}: start={start} limit={limit}")

            process: subprocess.CompletedProcess[str] | None = None
            try:
                process = run_metric_batch(args, metric_name, start, limit, csv_path, json_path)
                if process.returncode != 0:
                    raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "Subprocess failed.")

                batch_rows = load_batch_results(json_path)
                if not batch_rows:
                    raise ValueError(f"Empty batch results for metric {metric_name}: {json_path}")

                metric_rows.extend(batch_rows)
                metric_batch_records.append(
                    {
                        "metric": metric_name,
                        "start": start,
                        "limit": limit,
                        "end": start + limit - 1,
                        "status": "success",
                        "csv_output": str(csv_path.resolve()),
                        "json_output": str(json_path.resolve()),
                        "rows_written": len(batch_rows),
                    }
                )
                successful_batches += 1
            except Exception as exc:
                failed_batches += 1
                error_message = str(exc)
                write_error_batch_csv(csv_path, metric_name, start, limit, error_message)
                write_error_batch_json(json_path, metric_name, start, limit, process, error_message)
                metric_batch_records.append(
                    {
                        "metric": metric_name,
                        "start": start,
                        "limit": limit,
                        "end": start + limit - 1,
                        "status": "failed",
                        "error": error_message,
                        "csv_output": str(csv_path.resolve()),
                        "json_output": str(json_path.resolve()),
                        "rows_written": 0,
                    }
                )
                if not args.continue_on_error:
                    break

        merge_metric_rows(combined_rows, metric_rows, metric_name)
        merged_csv_path, merged_json_path = metric_merged_paths(output_dir, metric_name)
        metric_average_value = metric_average(metric_rows, metric_name)
        save_rows_csv(merged_csv_path, metric_rows)
        save_rows_json(
            merged_json_path,
            {
                "metric": metric_name,
                "summary": {
                    "rows_merged": len(metric_rows),
                    "average": metric_average_value,
                },
                "batches": metric_batch_records,
                "results": metric_rows,
            },
        )
        metric_summaries.append(
            {
                "metric": metric_name,
                "rows_merged": len(metric_rows),
                "average": metric_average_value,
                "csv_output": str(merged_csv_path.resolve()),
                "json_output": str(merged_json_path.resolve()),
            }
        )

        if not args.continue_on_error and failed_batches:
            break

    final_rows = [combined_rows[index] for index in sorted(combined_rows)]
    final_metric_averages = {
        metric_name: metric_average(final_rows, metric_name) for metric_name in metric_names
    }
    save_rows_csv(combined_csv_path, final_rows)
    save_rows_json(
        combined_json_path,
        {
            "run_config": {
                "gold_path": str(args.gold_path.resolve()),
                "total": args.total,
                "batch_size": args.batch_size,
                "metrics": args.metrics,
                "output_dir": str(output_dir),
                "combined_csv": str(combined_csv_path),
                "combined_json": str(combined_json_path),
                "continue_on_error": args.continue_on_error,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
                "embedding_model": args.embedding_model,
                "prompt_version": args.prompt_version,
                "answer_model": resolved_answer_model,
            },
            "summary": {
                "total_metrics": len(metric_names),
                "total_batches_attempted": total_batches_attempted,
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "rows_merged": len(final_rows),
                "metric_averages": final_metric_averages,
            },
            "metrics": metric_summaries,
            "results": final_rows,
        },
    )

    print("RAGAS metric-batch orchestration summary")
    print("=======================================")
    print(f"total metrics: {len(metric_names)}")
    print(f"total batches attempted: {total_batches_attempted}")
    print(f"successful batches: {successful_batches}")
    print(f"failed batches: {failed_batches}")
    print(f"rows merged: {len(final_rows)}")
    for metric_name, average in final_metric_averages.items():
        print(f"{metric_name}: {'unavailable' if average is None else f'{average:.4f}'}")
    print(f"Combined CSV: {combined_csv_path}")
    print(f"Combined JSON: {combined_json_path}")


if __name__ == "__main__":
    main()
