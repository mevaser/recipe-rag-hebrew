from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    EvaluationRunConfig,
    configure_stdout,
    execute_evaluation,
    flatten_row,
    load_jsonl,
    normalized_provider,
    parse_metric_names,
    resolve_answer_model_name,
    sanitize_model_name,
)


DEFAULT_BATCH_SIZE = 5
DEFAULT_TOTAL = 50
DEFAULT_BATCH_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ragas_batches"


def parse_args() -> argparse.Namespace:
    env_provider = normalized_provider(os.getenv("RAGAS_LLM_PROVIDER", "ollama"))
    parser = argparse.ArgumentParser(description="Run optional batched RAGAS evaluation.")
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
        help="Maximum number of examples to evaluate across all batches. Defaults to 50.",
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
        help="Directory for per-batch outputs. Defaults to a prompt-version-safe directory.",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="Path for combined CSV output. Defaults to a prompt-version-safe filename.",
    )
    parser.add_argument(
        "--combined-json",
        type=Path,
        default=None,
        help="Path for combined JSON output. Defaults to a prompt-version-safe filename.",
    )
    parser.add_argument(
        "--continue-on-error",
        type=parse_bool,
        default=True,
        help="Continue to later batches after a batch failure. Defaults to true.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=env_provider,
        help="Evaluator provider. Defaults to ollama.",
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
        help="Embedding model for local evaluation.",
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


def default_batch_output_dir(prompt_version: str, answer_model: str) -> Path:
    normalized = prompt_version.strip().lower()
    if normalized == "baseline":
        return DEFAULT_BATCH_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR / f"ragas_batches_{normalized}_{sanitize_model_name(answer_model)}"


def default_combined_csv_path(prompt_version: str, answer_model: str) -> Path:
    normalized = prompt_version.strip().lower()
    if normalized == "baseline":
        return DEFAULT_OUTPUT_DIR / "ragas_results_50.csv"
    return DEFAULT_OUTPUT_DIR / f"ragas_results_50_{normalized}_{sanitize_model_name(answer_model)}.csv"


def default_combined_json_path(prompt_version: str, answer_model: str) -> Path:
    normalized = prompt_version.strip().lower()
    if normalized == "baseline":
        return DEFAULT_OUTPUT_DIR / "ragas_results_50.json"
    return DEFAULT_OUTPUT_DIR / f"ragas_results_50_{normalized}_{sanitize_model_name(answer_model)}.json"


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


def batch_stem(start: int, limit: int) -> str:
    end = start + limit - 1
    return f"ragas_batch_{start:03d}_{end:03d}"


def batch_paths(output_dir: Path, start: int, limit: int) -> tuple[Path, Path]:
    stem = batch_stem(start, limit)
    return output_dir / f"{stem}.csv", output_dir / f"{stem}.json"


def error_row(start: int, limit: int, metric_names: list[str], error_message: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "batch_start": start,
        "batch_limit": limit,
        "batch_end": start + limit - 1,
        "batch_status": "failed",
        "error": error_message,
        "question_number": None,
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
    }
    for metric_name in metric_names:
        row[metric_name] = None
    return row


def enrich_success_rows(rows: list[dict[str, Any]], start: int, limit: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["batch_start"] = start
        enriched_row["batch_limit"] = limit
        enriched_row["batch_end"] = start + limit - 1
        enriched_row["batch_status"] = "success"
        enriched_row["error"] = ""
        enriched.append(enriched_row)
    return enriched


def save_error_batch_outputs(
    csv_path: Path,
    json_path: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(flatten_row(row).keys()))
        writer.writeheader()
        writer.writerow(flatten_row(row))

    payload = {
        "run_config": {
            "gold_path": str(args.gold_path.resolve()),
            "total": args.total,
            "batch_size": args.batch_size,
            "metrics": args.metrics,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "prompt_version": args.prompt_version,
            "answer_model": resolve_answer_model_name(args.answer_model),
        },
        "summary": {
            "examples_evaluated": 0,
            "metric_averages": {},
            "batch_status": "failed",
        },
        "results": [row],
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def combined_metric_averages(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float | None]:
    averages: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if isinstance(row.get(metric_name), (int, float)) and not math.isnan(float(row[metric_name]))
        ]
        averages[metric_name] = sum(values) / len(values) if values else None
    return averages


def save_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened_rows = [flatten_row(row) for row in rows]
    fieldnames = list(flattened_rows[0].keys()) if flattened_rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_rows)


def save_combined_json(
    path: Path,
    args: argparse.Namespace,
    batch_results: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    metric_averages: dict[str, float | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_config": {
            "gold_path": str(args.gold_path.resolve()),
            "total": args.total,
            "batch_size": args.batch_size,
            "metrics": args.metrics,
            "output_dir": str((args.output_dir.resolve() if args.output_dir else default_batch_output_dir(args.prompt_version).resolve())),
            "combined_csv": str((args.combined_csv.resolve() if args.combined_csv else default_combined_csv_path(args.prompt_version).resolve())),
            "combined_json": str((args.combined_json.resolve() if args.combined_json else default_combined_json_path(args.prompt_version).resolve())),
            "continue_on_error": args.continue_on_error,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "prompt_version": args.prompt_version,
        },
        "summary": {
            "total_batches": len(batch_results),
            "successful_batches": sum(1 for batch in batch_results if batch["status"] == "success"),
            "failed_batches": sum(1 for batch in batch_results if batch["status"] == "failed"),
            "combined_rows_written": len(combined_rows),
            "metric_averages": metric_averages,
        },
        "batches": batch_results,
        "results": combined_rows,
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def print_summary(batch_results: list[dict[str, Any]], combined_rows: list[dict[str, Any]], metric_averages: dict[str, float | None]) -> None:
    print("Batched RAGAS evaluation summary")
    print("================================")
    print(f"total batches: {len(batch_results)}")
    print(f"successful batches: {sum(1 for batch in batch_results if batch['status'] == 'success')}")
    print(f"failed batches: {sum(1 for batch in batch_results if batch['status'] == 'failed')}")
    print(f"combined rows written: {len(combined_rows)}")
    for metric_name, average in metric_averages.items():
        print(f"{metric_name}: {'unavailable' if average is None else f'{average:.4f}'}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    resolved_answer_model = resolve_answer_model_name(args.answer_model)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else default_batch_output_dir(args.prompt_version, resolved_answer_model).resolve()
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

    batch_results: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    for batch_index, (start, limit) in enumerate(batch_ranges, start=1):
        csv_path, json_path = batch_paths(output_dir, start, limit)
        print(f"Batch {batch_index}/{len(batch_ranges)}: start={start} limit={limit}")
        config = EvaluationRunConfig(
            gold_path=args.gold_path.resolve(),
            start=start,
            limit=limit,
            output_dir=output_dir,
            output_stem=batch_stem(start, limit),
            metrics=args.metrics,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            prompt_version=args.prompt_version,
            answer_model=resolved_answer_model,
        )

        try:
            artifacts = execute_evaluation(config, print_diagnostics=False)
            batch_rows = enrich_success_rows(artifacts.rows, start, limit)
            combined_rows.extend(batch_rows)
            batch_results.append(
                {
                    "batch_index": batch_index,
                    "start": start,
                    "limit": limit,
                    "end": start + limit - 1,
                    "status": "success",
                    "error": "",
                    "csv_output": str(artifacts.csv_output_path),
                    "json_output": str(artifacts.json_output_path),
                    "rows_written": len(batch_rows),
                }
            )
        except Exception as exc:
            failure_message = str(exc)
            failure_row = error_row(start, limit, metric_names, failure_message)
            save_error_batch_outputs(csv_path, json_path, failure_row, args)
            combined_rows.append(failure_row)
            batch_results.append(
                {
                    "batch_index": batch_index,
                    "start": start,
                    "limit": limit,
                    "end": start + limit - 1,
                    "status": "failed",
                    "error": failure_message,
                    "csv_output": str(csv_path),
                    "json_output": str(json_path),
                    "rows_written": 1,
                }
            )
            if not args.continue_on_error:
                break

    metric_averages = combined_metric_averages(combined_rows, metric_names)
    save_combined_csv(combined_csv_path, combined_rows)
    save_combined_json(combined_json_path, args, batch_results, combined_rows, metric_averages)
    print_summary(batch_results, combined_rows, metric_averages)
    print(f"Combined CSV: {combined_csv_path}")
    print(f"Combined JSON: {combined_json_path}")


if __name__ == "__main__":
    main()
