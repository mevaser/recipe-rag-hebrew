from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_retrieval import (  # noqa: E402
    DEFAULT_RERANK_BM25_WEIGHT,
    DEFAULT_RERANK_KEYWORD_WEIGHT,
    DEFAULT_RERANK_SOURCE_WEIGHT,
    retrieve_hybrid,
)


DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "lightweight_rerank_tuning_results.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "eval" / "lightweight_rerank_tuning_summary.csv"
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "eval" / "lightweight_rerank_tuning_findings.md"
QUESTION_NUMBERS = ["4", "5", "10", "35", "46", "29", "31", "41", "50"]
CONFIGS = [
    {
        "config_name": "bm25_only",
        "enable_lightweight_rerank": True,
        "rerank_bm25_weight": 0.30,
        "rerank_source_weight": 0.00,
        "rerank_keyword_weight": 0.00,
    },
    {
        "config_name": "bm25_keyword",
        "enable_lightweight_rerank": True,
        "rerank_bm25_weight": 0.30,
        "rerank_source_weight": 0.00,
        "rerank_keyword_weight": 0.10,
    },
    {
        "config_name": "conservative_source",
        "enable_lightweight_rerank": True,
        "rerank_bm25_weight": 0.25,
        "rerank_source_weight": 0.05,
        "rerank_keyword_weight": 0.10,
    },
    {
        "config_name": "current_default",
        "enable_lightweight_rerank": True,
        "rerank_bm25_weight": DEFAULT_RERANK_BM25_WEIGHT,
        "rerank_source_weight": DEFAULT_RERANK_SOURCE_WEIGHT,
        "rerank_keyword_weight": DEFAULT_RERANK_KEYWORD_WEIGHT,
    },
    {
        "config_name": "no_rerank_baseline",
        "enable_lightweight_rerank": False,
        "rerank_bm25_weight": 0.0,
        "rerank_source_weight": 0.0,
        "rerank_keyword_weight": 0.0,
    },
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune lightweight rerank weights with control-question guardrails.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Detailed output CSV path.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH, help="Summary CSV path.")
    parser.add_argument("--findings-output", type=Path, default=DEFAULT_FINDINGS_PATH, help="Markdown findings path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def top_sources(results: list[dict[str, Any]], top_n: int = 5) -> list[str]:
    values: list[str] = []
    for result in results[:top_n]:
        metadata = result.get("metadata", {})
        source = normalize_text(metadata.get("source", "")) or normalize_text(metadata.get("relative_path", ""))
        chunk_id = normalize_text(result.get("chunk_id", ""))
        values.append(f"{source} [{chunk_id}]")
    return values


def rank_of_expected(results: list[dict[str, Any]], expected_chunk_id: str) -> int | None:
    for index, result in enumerate(results, start=1):
        if normalize_text(result.get("chunk_id", "")) == expected_chunk_id:
            return index
    return None


def hit_at_k(rank: int | None, k: int) -> str:
    return "yes" if rank is not None and rank <= k else "no"


def expected_source_from_chunk_id(chunk_id: str) -> str:
    if "_chunk_" in chunk_id:
        return chunk_id.rsplit("_chunk_", 1)[0]
    return chunk_id


def format_rank_delta(rank_delta: int | None) -> str:
    if rank_delta is None:
        return ""
    return str(rank_delta)


def check_output_paths(paths: list[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")


def build_markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = [
        "config_name",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "improved_count",
        "worsened_count",
        "avg_rank_delta",
        "worsened_questions",
        "q41_worsened",
        "q50_rank_delta",
        "recommended",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_findings_markdown(summary_frame: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    recommended_rows = summary_frame.loc[summary_frame["recommended"] == "yes", "config_name"].tolist()
    if recommended_rows:
        recommendation = (
            f"The safest configuration in this diagnostic is `{recommended_rows[0]}` because it improved rank ordering "
            "without worsening Q41 or Q50."
        )
    else:
        recommendation = "No tested configuration was safe enough. Keep lightweight rerank diagnostic-only for now."

    lines = [
        "# Lightweight Rerank Tuning Findings",
        "",
        "## Goal",
        "Tune the rerank layer without changing embeddings, chunking, or the index.",
        "",
        "## Why Tuning Was Needed",
        "The first rerank improved Q5, Q10, and Q46 but worsened Q41, so the weights need guardrail-based tuning.",
        "",
        "## Configurations Tested",
        "- Config A: `bm25_only`",
        "- Config B: `bm25_keyword`",
        "- Config C: `conservative_source`",
        "- Config D: `current_default`",
        "- Config E: `no_rerank_baseline`",
        "",
        "## Summary Table",
        *build_markdown_table(summary_frame),
        "",
        "## Recommendation",
        recommendation,
        "",
        "## Next Steps",
        "- If a safe config exists, test it on a small generation subset.",
        "- If no safe config exists, consider a neural reranker later.",
        "- Continue with Hebrew/RTL normalization and the GPT backend decision.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_stdout()
    args = parse_args()
    check_output_paths([args.output, args.summary_output, args.findings_output], overwrite=args.overwrite)

    gold_rows = load_gold_rows(DEFAULT_GOLD_PATH)
    gold_lookup = {
        str(index): row
        for index, row in enumerate(gold_rows, start=1)
        if str(index) in QUESTION_NUMBERS
    }

    baseline_result_cache: dict[str, list[dict[str, Any]]] = {}
    for question_number in QUESTION_NUMBERS:
        question = normalize_text(gold_lookup[question_number].get("question", ""))
        baseline_result_cache[question_number] = retrieve_hybrid(question, k=100, candidate_k=100)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for config in CONFIGS:
        config_name = config["config_name"]
        config_detail_rows: list[dict[str, Any]] = []
        improved_count = 0
        worsened_count = 0
        rank_deltas: list[int] = []
        worsened_questions: list[str] = []
        hit_at_1_count = 0
        hit_at_3_count = 0
        hit_at_5_count = 0
        q41_worsened = False
        q50_rank_delta: int | None = None

        for question_number in QUESTION_NUMBERS:
            gold_row = gold_lookup[question_number]
            question = normalize_text(gold_row.get("question", ""))
            expected_chunk_id = normalize_text((gold_row.get("must_cite_chunk_ids") or [""])[0])

            baseline_results = baseline_result_cache[question_number]
            baseline_rank = rank_of_expected(baseline_results, expected_chunk_id)

            if config["enable_lightweight_rerank"]:
                reranked_results = retrieve_hybrid(
                    question,
                    k=100,
                    candidate_k=100,
                    enable_lightweight_rerank=True,
                    rerank_bm25_weight=float(config["rerank_bm25_weight"]),
                    rerank_source_weight=float(config["rerank_source_weight"]),
                    rerank_keyword_weight=float(config["rerank_keyword_weight"]),
                )
            else:
                reranked_results = baseline_results

            reranked_rank = rank_of_expected(reranked_results, expected_chunk_id)
            rank_delta = None
            if baseline_rank is not None and reranked_rank is not None:
                rank_delta = baseline_rank - reranked_rank
                rank_deltas.append(rank_delta)

            improved = rank_delta is not None and rank_delta > 0
            worsened = rank_delta is not None and rank_delta < 0
            if improved:
                improved_count += 1
            if worsened:
                worsened_count += 1
                worsened_questions.append(f"Q{question_number}")
            if question_number == "41" and worsened:
                q41_worsened = True
            if question_number == "50":
                q50_rank_delta = rank_delta

            reranked_hit_at_1 = hit_at_k(reranked_rank, 1)
            reranked_hit_at_3 = hit_at_k(reranked_rank, 3)
            reranked_hit_at_5 = hit_at_k(reranked_rank, 5)
            if reranked_hit_at_1 == "yes":
                hit_at_1_count += 1
            if reranked_hit_at_3 == "yes":
                hit_at_3_count += 1
            if reranked_hit_at_5 == "yes":
                hit_at_5_count += 1

            detail_row = {
                "config_name": config_name,
                "question_number": question_number,
                "baseline_expected_rank": "" if baseline_rank is None else baseline_rank,
                "reranked_expected_rank": "" if reranked_rank is None else reranked_rank,
                "baseline_hit_at_1": hit_at_k(baseline_rank, 1),
                "reranked_hit_at_1": reranked_hit_at_1,
                "baseline_hit_at_3": hit_at_k(baseline_rank, 3),
                "reranked_hit_at_3": reranked_hit_at_3,
                "baseline_hit_at_5": hit_at_k(baseline_rank, 5),
                "reranked_hit_at_5": reranked_hit_at_5,
                "rank_delta": format_rank_delta(rank_delta),
                "improved": "yes" if improved else "no",
                "worsened": "yes" if worsened else "no",
                "baseline_top_5_sources": json.dumps(top_sources(baseline_results), ensure_ascii=False),
                "reranked_top_5_sources": json.dumps(top_sources(reranked_results), ensure_ascii=False),
                "notes": "",
            }
            detail_rows.append(detail_row)
            config_detail_rows.append(detail_row)

        avg_rank_delta = round(sum(rank_deltas) / len(rank_deltas), 4) if rank_deltas else 0.0
        q50_delta_text = "" if q50_rank_delta is None else str(q50_rank_delta)
        recommended = (
            config_name != "no_rerank_baseline"
            and improved_count > 0
            and worsened_count == 0
            and not q41_worsened
            and (q50_rank_delta is None or q50_rank_delta >= 0)
        )
        summary_rows.append(
            {
                "config_name": config_name,
                "hit_at_1": hit_at_1_count,
                "hit_at_3": hit_at_3_count,
                "hit_at_5": hit_at_5_count,
                "improved_count": improved_count,
                "worsened_count": worsened_count,
                "avg_rank_delta": avg_rank_delta,
                "worsened_questions": ",".join(worsened_questions),
                "q41_worsened": "yes" if q41_worsened else "no",
                "q50_rank_delta": q50_delta_text,
                "recommended": "yes" if recommended else "no",
            }
        )

    detail_frame = pd.DataFrame(detail_rows)
    summary_frame = pd.DataFrame(summary_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary_frame.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    build_findings_markdown(summary_frame, args.findings_output, overwrite=args.overwrite)

    best_config = summary_frame.loc[summary_frame["recommended"] == "yes", "config_name"].tolist()
    best_config_text = best_config[0] if best_config else "none"
    print(f"Questions tested: {len(QUESTION_NUMBERS)}")
    print(f"Detailed results CSV: {args.output}")
    print(f"Summary CSV: {args.summary_output}")
    print(f"Findings Markdown: {args.findings_output}")
    print(f"Best config: {best_config_text}")
    if best_config:
        best_row = summary_frame.loc[summary_frame["config_name"] == best_config[0]].iloc[0]
        print(f"Best config Q41 worsened: {best_row['q41_worsened']}")
        q50_delta = str(best_row["q50_rank_delta"]).strip()
        print(f"Best config Q50 rank delta: {q50_delta or 'none'}")
    else:
        print("Best config Q41 worsened: n/a")
        print("Best config Q50 rank delta: n/a")


if __name__ == "__main__":
    main()
