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

from hybrid_retrieval import retrieve_hybrid  # noqa: E402


DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
DEFAULT_MISSING_CONTEXT_PATH = PROJECT_ROOT / "eval" / "missing_context_investigation.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "lightweight_rerank_comparison.csv"
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "eval" / "lightweight_rerank_findings.md"
QUESTION_NUMBERS = ["4", "5", "10", "35", "46", "29", "31", "41", "50"]
RERANK_BM25_WEIGHT = 0.30
RERANK_SOURCE_WEIGHT = 0.20
RERANK_KEYWORD_WEIGHT = 0.10


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


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


def normalize_question_number(value: Any) -> str:
    text = normalize_text(value)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline hybrid retrieval against lightweight reranking.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--findings-output",
        type=Path,
        default=DEFAULT_FINDINGS_PATH,
        help="Output Markdown findings path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def expected_source_from_chunk_id(chunk_id: str) -> str:
    if "_chunk_" in chunk_id:
        return chunk_id.rsplit("_chunk_", 1)[0]
    return chunk_id


def rank_of_expected(results: list[dict[str, Any]], expected_chunk_id: str) -> int | None:
    for index, result in enumerate(results, start=1):
        if normalize_text(result.get("chunk_id", "")) == expected_chunk_id:
            return index
    return None


def top_sources(results: list[dict[str, Any]], top_n: int = 5) -> list[str]:
    values: list[str] = []
    for result in results[:top_n]:
        metadata = result.get("metadata", {})
        source = normalize_text(metadata.get("source", "")) or normalize_text(metadata.get("relative_path", ""))
        chunk_id = normalize_text(result.get("chunk_id", ""))
        values.append(f"{source} [{chunk_id}]")
    return values


def build_findings_markdown(
    output_path: Path,
    baseline_hit_at_5: int,
    reranked_hit_at_5: int,
    improved_questions: list[str],
    worsened_questions: list[str],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")
    lines = [
        "# Lightweight Rerank Findings",
        "",
        "## Goal",
        "Improve ranking for cases where expected chunks exist but are ranked too low.",
        "",
        "## Method",
        "No embeddings, chunking, or index rebuild was changed. The diagnostic compared baseline hybrid retrieval against an optional lightweight rerank layer that adds BM25/source/keyword boosts after hybrid fusion.",
        "",
        "## Questions Tested",
        "Q4, Q5, Q10, Q35, Q46 plus controls Q29, Q31, Q41, Q50.",
        "",
        "## Results",
        f"- baseline Hit@5: {baseline_hit_at_5}/9",
        f"- reranked Hit@5: {reranked_hit_at_5}/9",
        f"- improved questions: {', '.join(improved_questions) if improved_questions else 'none'}",
        f"- worsened questions: {', '.join(worsened_questions) if worsened_questions else 'none'}",
        "",
        "## Interpretation",
        "The lightweight rerank layer shows whether simple BM25/source/keyword boosts can improve expected chunk ranking without changing embeddings or chunking.",
        "",
        "## Next Steps",
        "- If rerank improves Hit@5 without hurting controls, consider enabling it for evaluation.",
        "- If rerank hurts controls, tune weights or keep it diagnostic only.",
        "- If rerank is insufficient, consider neural reranker.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_stdout()
    args = parse_args()
    gold_rows = load_gold_rows(DEFAULT_GOLD_PATH)
    missing_context_frame = pd.read_csv(DEFAULT_MISSING_CONTEXT_PATH, encoding="utf-8-sig")
    missing_reason_lookup = {
        normalize_question_number(row["question_number"]): normalize_text(row["suspected_reason"])
        for _, row in missing_context_frame.iterrows()
    }

    rows: list[dict[str, Any]] = []
    improved_questions: list[str] = []
    worsened_questions: list[str] = []
    baseline_hits = 0
    reranked_hits = 0

    for question_number in QUESTION_NUMBERS:
        gold_row = gold_rows[int(question_number) - 1]
        question = normalize_text(gold_row.get("question", ""))
        reference = normalize_text(gold_row.get("reference_answer", ""))
        expected_chunk_id = normalize_text((gold_row.get("must_cite_chunk_ids") or [""])[0])
        expected_source = expected_source_from_chunk_id(expected_chunk_id)

        baseline_results = retrieve_hybrid(question, k=100, candidate_k=100)
        reranked_results = retrieve_hybrid(
            question,
            k=100,
            candidate_k=100,
            enable_lightweight_rerank=True,
            rerank_bm25_weight=RERANK_BM25_WEIGHT,
            rerank_source_weight=RERANK_SOURCE_WEIGHT,
            rerank_keyword_weight=RERANK_KEYWORD_WEIGHT,
        )

        baseline_rank = rank_of_expected(baseline_results, expected_chunk_id)
        reranked_rank = rank_of_expected(reranked_results, expected_chunk_id)
        baseline_hit = baseline_rank is not None and baseline_rank <= 5
        reranked_hit = reranked_rank is not None and reranked_rank <= 5
        improved = (
            baseline_rank is None
            or (reranked_rank is not None and reranked_rank < baseline_rank)
        )
        worsened = (
            baseline_rank is not None
            and reranked_rank is not None
            and reranked_rank > baseline_rank
        ) or (baseline_hit and not reranked_hit)

        if baseline_hit:
            baseline_hits += 1
        if reranked_hit:
            reranked_hits += 1
        if improved and not worsened and baseline_rank != reranked_rank:
            improved_questions.append(f"Q{question_number}")
        if worsened:
            worsened_questions.append(f"Q{question_number}")

        rows.append(
            {
                "question_number": question_number,
                "question": question,
                "reference": reference,
                "expected_source": expected_source,
                "baseline_expected_rank": "" if baseline_rank is None else baseline_rank,
                "reranked_expected_rank": "" if reranked_rank is None else reranked_rank,
                "baseline_hit_at_5": "yes" if baseline_hit else "no",
                "reranked_hit_at_5": "yes" if reranked_hit else "no",
                "baseline_top_5_sources": json.dumps(top_sources(baseline_results), ensure_ascii=False),
                "reranked_top_5_sources": json.dumps(top_sources(reranked_results), ensure_ascii=False),
                "improved": "yes" if improved and not worsened and baseline_rank != reranked_rank else "no",
                "suspected_reason": missing_reason_lookup.get(question_number, "control_question"),
                "manual_notes": "",
            }
        )

    output_frame = pd.DataFrame(rows)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {args.output}. Use --overwrite to replace it.")
    if args.findings_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.findings_output}. Use --overwrite to replace it."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output, index=False, encoding="utf-8-sig")

    build_findings_markdown(
        args.findings_output,
        baseline_hit_at_5=baseline_hits,
        reranked_hit_at_5=reranked_hits,
        improved_questions=improved_questions,
        worsened_questions=worsened_questions,
        overwrite=args.overwrite,
    )

    print(f"Questions tested: {len(rows)}")
    print(f"Baseline Hit@5 for expected chunks: {baseline_hits}/{len(rows)}")
    print(f"Reranked Hit@5 for expected chunks: {reranked_hits}/{len(rows)}")
    print(f"Questions improved: {len(improved_questions)}")
    print(f"Questions worsened: {len(worsened_questions)}")
    if worsened_questions:
        print("Worsened questions: " + ", ".join(worsened_questions))
    else:
        print("Worsened questions: none")
    print(f"Output CSV: {args.output}")
    print(f"Findings Markdown: {args.findings_output}")


if __name__ == "__main__":
    main()
