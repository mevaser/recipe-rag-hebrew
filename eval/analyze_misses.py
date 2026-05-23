from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MISSES_PATH = PROJECT_ROOT / "eval" / "misses_at_20.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "eval" / "eval_results.json"
GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
OUTPUT_JSON_PATH = PROJECT_ROOT / "eval" / "failure_analysis.json"
OUTPUT_MD_PATH = PROJECT_ROOT / "eval" / "failure_analysis.md"
PREVIEW_CHARS = 500

GENERIC_QUESTION_TERMS = {
    "כמה",
    "מה",
    "איך",
    "איזה",
    "אילו",
    "למה",
    "מתי",
    "האם",
    "צריך",
    "צריכים",
    "עושים",
    "מכינים",
    "מתכון",
    "במתכון",
    "שלב",
    "אחרי",
    "לפני",
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {path}.")
            rows.append(row)
    return rows


def chunk_lookup(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}


def text_preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def chunk_preview(chunk_id: str, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chunk = lookup.get(chunk_id)
    if chunk is None:
        return {
            "chunk_id": chunk_id,
            "found": False,
            "source": None,
            "text_preview": None,
        }

    metadata = chunk.get("metadata", {})
    return {
        "chunk_id": chunk_id,
        "found": True,
        "source": metadata.get("source", ""),
        "category": metadata.get("category", ""),
        "page": metadata.get("page"),
        "text_preview": text_preview(str(chunk.get("text", ""))),
    }


def source_tokens(source: str) -> set[str]:
    stem = Path(source).stem
    tokens = re.findall(r"[\u0590-\u05ffA-Za-z0-9]+", stem.casefold())
    return {token for token in tokens if len(token) > 1 and token not in GENERIC_QUESTION_TERMS}


def question_tokens(question: str) -> set[str]:
    tokens = re.findall(r"[\u0590-\u05ffA-Za-z0-9]+", question.casefold())
    return {token for token in tokens if len(token) > 1 and token not in GENERIC_QUESTION_TERMS}


def expected_sources(expected_chunk_ids: list[str], lookup: dict[str, dict[str, Any]]) -> set[str]:
    sources: set[str] = set()
    for chunk_id in expected_chunk_ids:
        chunk = lookup.get(chunk_id)
        if chunk:
            source = str(chunk.get("metadata", {}).get("source", ""))
            if source:
                sources.add(source)
    return sources


def retrieved_sources(retrieved_chunk_ids: list[str], lookup: dict[str, dict[str, Any]]) -> set[str]:
    sources: set[str] = set()
    for chunk_id in retrieved_chunk_ids:
        chunk = lookup.get(chunk_id)
        if chunk:
            source = str(chunk.get("metadata", {}).get("source", ""))
            if source:
                sources.add(source)
    return sources


def is_generic_question(question: str, expected_source_names: set[str]) -> bool:
    tokens = question_tokens(question)
    if len(tokens) < 3:
        return True

    source_terms: set[str] = set()
    for source in expected_source_names:
        source_terms.update(source_tokens(source))

    if not source_terms:
        return False
    return not bool(tokens & source_terms)


def guess_failure_type(miss: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> tuple[str, str]:
    expected_chunk_ids = [str(chunk_id) for chunk_id in miss.get("expected_chunk_ids", [])]
    retrieved_chunk_ids = [str(chunk_id) for chunk_id in miss.get("retrieved_chunk_ids", [])]

    missing_expected = [chunk_id for chunk_id in expected_chunk_ids if chunk_id not in lookup]
    if missing_expected:
        return "chunking_problem", "At least one expected chunk ID is missing from indexed_chunks.json."

    expected_source_names = expected_sources(expected_chunk_ids, lookup)
    retrieved_source_names = retrieved_sources(retrieved_chunk_ids, lookup)
    if expected_source_names & retrieved_source_names:
        return "expected_chunk_too_strict", "Retrieved chunks include the expected source but not the exact expected chunk ID."

    if is_generic_question(str(miss.get("question", "")), expected_source_names):
        return "bad_gold_question", "Question appears generic or does not mention terms from the expected recipe/source name."

    return "unclear_needs_manual_review", "No deterministic heuristic identified the failure cause."


def analyze_miss(miss: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_chunk_ids = [str(chunk_id) for chunk_id in miss.get("expected_chunk_ids", [])]
    retrieved_chunk_ids = [str(chunk_id) for chunk_id in miss.get("retrieved_chunk_ids", [])]
    failure_type, notes = guess_failure_type(miss, lookup)

    return {
        "question_number": miss.get("question_number"),
        "question": miss.get("question", ""),
        "category": miss.get("category", ""),
        "expected_chunk_ids": expected_chunk_ids,
        "expected_chunk_text_preview": [
            chunk_preview(chunk_id, lookup)
            for chunk_id in expected_chunk_ids
        ],
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunk_text_previews": [
            chunk_preview(chunk_id, lookup)
            for chunk_id in retrieved_chunk_ids
        ],
        "failure_type_guess": failure_type,
        "notes": notes,
    }


def summarize_failures(analyses: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for analysis in analyses:
        failure_type = str(analysis["failure_type_guess"])
        summary[failure_type] = summary.get(failure_type, 0) + 1
    return summary


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def markdown_chunk_list(previews: list[dict[str, Any]], max_items: int | None = None) -> str:
    selected = previews[:max_items] if max_items is not None else previews
    lines: list[str] = []
    for preview in selected:
        lines.append(f"- `{preview['chunk_id']}`")
        lines.append(f"  - Found: {preview['found']}")
        lines.append(f"  - Source: {preview.get('source')}")
        lines.append(f"  - Preview: {preview.get('text_preview')}")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Retrieval Failure Analysis",
        "",
        f"Misses analyzed: {report['summary']['misses_analyzed']}",
        "",
        "## Failure Type Counts",
        "",
    ]
    for failure_type, count in report["summary"]["failure_type_counts"].items():
        lines.append(f"- `{failure_type}`: {count}")

    lines.extend(["", "## Miss Details", ""])
    for analysis in report["misses"]:
        lines.extend(
            [
                f"### Question {analysis['question_number']}",
                "",
                f"- Category: `{analysis['category']}`",
                f"- Failure type guess: `{analysis['failure_type_guess']}`",
                f"- Notes: {analysis['notes']}",
                f"- Question: {analysis['question']}",
                "",
                "Expected chunks:",
                markdown_chunk_list(analysis["expected_chunk_text_preview"]),
                "",
                "Top retrieved chunks:",
                markdown_chunk_list(analysis["retrieved_chunk_text_previews"], max_items=5),
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    misses = load_json(MISSES_PATH)
    eval_results = load_json(EVAL_RESULTS_PATH)
    gold_rows = load_jsonl(GOLD_PATH)
    chunks = load_json(INDEXED_CHUNKS_PATH)
    lookup = chunk_lookup(chunks)

    if not isinstance(misses, list):
        raise ValueError(f"Expected a list of misses in {MISSES_PATH}.")
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {INDEXED_CHUNKS_PATH}.")

    analyses = [analyze_miss(miss, lookup) for miss in misses]
    return {
        "inputs": {
            "misses_path": str(MISSES_PATH),
            "eval_results_path": str(EVAL_RESULTS_PATH),
            "gold_path": str(GOLD_PATH),
            "indexed_chunks_path": str(INDEXED_CHUNKS_PATH),
        },
        "summary": {
            "misses_analyzed": len(analyses),
            "total_gold_questions": len(gold_rows),
            "eval_hit_at_20": eval_results.get("hit_curve", {}).get("hit_at_20", {}),
            "failure_type_counts": summarize_failures(analyses),
        },
        "misses": analyses,
    }


def main() -> None:
    configure_stdout()
    report = build_report()
    write_json(report, OUTPUT_JSON_PATH)
    write_markdown(report, OUTPUT_MD_PATH)
    print("Failure analysis summary")
    print("========================")
    print(f"Misses analyzed: {report['summary']['misses_analyzed']}")
    for failure_type, count in report["summary"]["failure_type_counts"].items():
        print(f"{failure_type}: {count}")
    print(f"Failure analysis JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"Failure analysis Markdown saved to: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
