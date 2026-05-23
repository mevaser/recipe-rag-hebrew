from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bm25_retrieval import retrieve_bm25  # noqa: E402
from hybrid_retrieval import retrieve_hybrid  # noqa: E402
from retrieval import retrieve as retrieve_dense  # noqa: E402


DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "eval" / "query_diagnostics.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "eval" / "query_diagnostics.md"
DEFAULT_INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_K = 30
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 2.0
TOKEN_RE = re.compile(r"[\u0590-\u05ffA-Za-z0-9]+")
GENERIC_TERMS = {
    "כמה",
    "מה",
    "איך",
    "איזה",
    "אילו",
    "צריך",
    "צריכים",
    "למתכון",
    "במתכון",
    "מתכון",
    "התבשיל",
    "התפחה",
    "השנייה",
    "הנחיות",
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose retrieval results for one query.")
    parser.add_argument("--query", required=True, help="Query text to diagnose.")
    parser.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
        help="Retrieval mode. Defaults to hybrid.",
    )
    parser.add_argument("--k", type=int, default=20, help="Number of results to inspect. Defaults to 20.")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def diagnostic_terms(query: str) -> list[str]:
    terms = []
    for token in tokenize(query):
        if len(token) < 2:
            continue
        if token in GENERIC_TERMS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def text_preview(text: str, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def matched_terms(terms: list[str], value: str) -> list[str]:
    value_tokens = set(tokenize(value))
    return [term for term in terms if term in value_tokens]


def retrieve_for_mode(query: str, mode: str, k: int) -> list[dict[str, Any]]:
    if mode == "dense":
        return retrieve_dense(query, k=k)
    if mode == "bm25":
        return retrieve_bm25(query, k=k)
    return retrieve_hybrid(
        query,
        k=k,
        candidate_k=max(k, DEFAULT_CANDIDATE_K),
        rrf_k=DEFAULT_RRF_K,
        dense_weight=DEFAULT_DENSE_WEIGHT,
        bm25_weight=DEFAULT_BM25_WEIGHT,
    )


def load_indexed_chunk_lookup(path: Path = DEFAULT_INDEXED_CHUNKS_PATH) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {path}.")
    return {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}


def diagnose_result(
    rank: int,
    result: dict[str, Any],
    terms: list[str],
    indexed_chunk_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    text = str(result.get("text", ""))
    indexed_chunk = indexed_chunk_lookup.get(str(result.get("chunk_id", "")), {})
    indexed_text = str(indexed_chunk.get("indexed_text", ""))
    source = str(metadata.get("source", ""))
    category = str(metadata.get("category", ""))
    source_category = f"{source} {category}"
    text_matches = matched_terms(terms, text)
    indexed_text_matches = matched_terms(terms, indexed_text)
    metadata_matches = matched_terms(terms, source_category)

    return {
        "rank": rank,
        "chunk_id": result.get("chunk_id", ""),
        "score": float(result.get("score", 0.0)),
        "source": source,
        "category": category,
        "text_preview": text_preview(text),
        "query_terms_in_text": bool(text_matches),
        "query_terms_in_indexed_text": bool(indexed_text_matches),
        "query_terms_in_source_or_category": bool(metadata_matches),
        "matched_text_terms": text_matches,
        "matched_indexed_text_terms": indexed_text_matches,
        "matched_source_or_category_terms": metadata_matches,
    }


def build_report(query: str, mode: str, k: int) -> dict[str, Any]:
    terms = diagnostic_terms(query)
    results = retrieve_for_mode(query, mode, k)
    indexed_chunk_lookup = load_indexed_chunk_lookup()
    diagnostics = [
        diagnose_result(rank, result, terms, indexed_chunk_lookup)
        for rank, result in enumerate(results, start=1)
    ]
    return {
        "query": query,
        "mode": mode,
        "k": k,
        "query_terms": terms,
        "results": diagnostics,
    }


def write_json(report: dict[str, Any], path: Path = DEFAULT_OUTPUT_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)


def markdown_terms(terms: list[str]) -> str:
    if not terms:
        return "None"
    return ", ".join(f"`{term}`" for term in terms)


def write_markdown(report: dict[str, Any], path: Path = DEFAULT_OUTPUT_MD) -> None:
    lines = [
        "# Query Diagnostics",
        "",
        f"- Query: {report['query']}",
        f"- Mode: `{report['mode']}`",
        f"- K: `{report['k']}`",
        f"- Query terms: {markdown_terms(report['query_terms'])}",
        "",
        "## Results",
        "",
    ]
    for result in report["results"]:
        lines.extend(
            [
                f"### Rank {result['rank']}",
                "",
                f"- Chunk ID: `{result['chunk_id']}`",
                f"- Score: `{result['score']:.6f}`",
                f"- Source: {result['source']}",
                f"- Category: {result['category']}",
                f"- Query terms in text: `{result['query_terms_in_text']}` ({markdown_terms(result['matched_text_terms'])})",
                (
                    "- Query terms in indexed_text: "
                    f"`{result['query_terms_in_indexed_text']}` "
                    f"({markdown_terms(result['matched_indexed_text_terms'])})"
                ),
                (
                    "- Query terms in source/category: "
                    f"`{result['query_terms_in_source_or_category']}` "
                    f"({markdown_terms(result['matched_source_or_category_terms'])})"
                ),
                f"- Preview: {result['text_preview']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print("Query diagnostics")
    print("=================")
    print(f"Query: {report['query']}")
    print(f"Mode: {report['mode']}")
    print(f"K: {report['k']}")
    print(f"Query terms: {report['query_terms']}")
    print()
    for result in report["results"]:
        print(f"Rank {result['rank']}: {result['chunk_id']}")
        print(f"  Score: {result['score']:.6f}")
        print(f"  Source: {result['source']}")
        print(f"  Category: {result['category']}")
        print(f"  Terms in text: {result['query_terms_in_text']} {result['matched_text_terms']}")
        print(f"  Terms in indexed_text: {result['query_terms_in_indexed_text']} {result['matched_indexed_text_terms']}")
        print(
            "  Terms in source/category: "
            f"{result['query_terms_in_source_or_category']} {result['matched_source_or_category_terms']}"
        )
        print(f"  Preview: {result['text_preview']}")
        print()
    print(f"JSON saved to: {DEFAULT_OUTPUT_JSON}")
    print(f"Markdown saved to: {DEFAULT_OUTPUT_MD}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    report = build_report(args.query, args.mode, args.k)
    write_json(report)
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
