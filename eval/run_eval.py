from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import retrieve as retrieve_dense  # noqa: E402


DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "eval" / "eval_results.json"
DEFAULT_MISSES_AT_5_PATH = PROJECT_ROOT / "eval" / "misses_at_5.json"
DEFAULT_MISSES_AT_10_PATH = PROJECT_ROOT / "eval" / "misses_at_10.json"
DEFAULT_MISSES_AT_20_PATH = PROJECT_ROOT / "eval" / "misses_at_20.json"
DEFAULT_INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
DEFAULT_MAX_K = 20
EVAL_K_VALUES = (1, 3, 5, 10, 20)
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_RERANK_BM25_WEIGHT = 0.30
DEFAULT_RERANK_SOURCE_WEIGHT = 0.20
DEFAULT_RERANK_KEYWORD_WEIGHT = 0.10


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval ranking quality against a JSONL gold set.")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Gold set JSONL path. Defaults to eval/gold_set.jsonl.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Legacy alias for max retrieval depth. The evaluator always computes Hit@1/3/5/10/20.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=DEFAULT_MAX_K,
        help="Maximum retrieved chunks per question. Defaults to 20.",
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
        help="Retrieval mode. Defaults to dense.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Candidate depth for hybrid dense and BM25 retrieval. Defaults to 50.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="Reciprocal Rank Fusion constant for hybrid retrieval. Defaults to 60.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Dense retrieval weight for hybrid RRF. Defaults to 1.0.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
        help="BM25 retrieval weight for hybrid RRF. Defaults to 1.0.",
    )
    parser.add_argument(
        "--enable-lightweight-rerank",
        action="store_true",
        help="Enable lightweight BM25/source/keyword reranking after hybrid fusion.",
    )
    parser.add_argument(
        "--rerank-bm25-weight",
        type=float,
        default=DEFAULT_RERANK_BM25_WEIGHT,
        help="Lightweight rerank BM25 rank weight. Defaults to 0.30.",
    )
    parser.add_argument(
        "--rerank-source-weight",
        type=float,
        default=DEFAULT_RERANK_SOURCE_WEIGHT,
        help="Lightweight rerank source/title overlap weight. Defaults to 0.20.",
    )
    parser.add_argument(
        "--rerank-keyword-weight",
        type=float,
        default=DEFAULT_RERANK_KEYWORD_WEIGHT,
        help="Lightweight rerank keyword overlap weight. Defaults to 0.10.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
            validate_example(example, line_number)
            examples.append(example)
    return examples


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def validate_example(example: dict, line_number: int) -> None:
    required_fields = {
        "question": str,
        "reference_answer": str,
        "must_cite_chunk_ids": list,
        "category": str,
    }
    for field, expected_type in required_fields.items():
        if field not in example:
            raise ValueError(f"Missing field '{field}' on gold set line {line_number}.")
        if not isinstance(example[field], expected_type):
            raise ValueError(f"Field '{field}' on gold set line {line_number} has the wrong type.")


def first_relevant_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> int | None:
    expected = set(expected_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return rank
    return None


def build_chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}


def chunk_source(chunk_id: str, chunk_lookup: dict[str, dict]) -> str | None:
    chunk = chunk_lookup.get(chunk_id)
    if not chunk:
        return None
    source = chunk.get("metadata", {}).get("source")
    return str(source) if source else None


def parse_chunk_index(chunk_id: str) -> int | None:
    match = re.search(r"_chunk_(\d+)$", chunk_id)
    if not match:
        return None
    return int(match.group(1))


def source_match(
    retrieved_chunk_id: str,
    expected_chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
) -> bool:
    retrieved_source = chunk_source(retrieved_chunk_id, chunk_lookup)
    if retrieved_source is None:
        return False
    expected_sources = {
        source
        for source in (chunk_source(chunk_id, chunk_lookup) for chunk_id in expected_chunk_ids)
        if source is not None
    }
    return retrieved_source in expected_sources


def neighbor_match(
    retrieved_chunk_id: str,
    expected_chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
) -> bool:
    if retrieved_chunk_id in set(expected_chunk_ids):
        return True

    retrieved_source = chunk_source(retrieved_chunk_id, chunk_lookup)
    retrieved_index = parse_chunk_index(retrieved_chunk_id)
    if retrieved_source is None or retrieved_index is None:
        return False

    for expected_chunk_id in expected_chunk_ids:
        expected_source = chunk_source(expected_chunk_id, chunk_lookup)
        expected_index = parse_chunk_index(expected_chunk_id)
        if expected_source is None or expected_index is None:
            continue
        if retrieved_source == expected_source and abs(retrieved_index - expected_index) <= 1:
            return True
    return False


def first_relevant_rank_by_match(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    match_type: str,
    chunk_lookup: dict[str, dict],
) -> int | None:
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if match_type == "source" and source_match(chunk_id, expected_chunk_ids, chunk_lookup):
            return rank
        if match_type == "neighbor" and neighbor_match(chunk_id, expected_chunk_ids, chunk_lookup):
            return rank
    return None


def hit_at_rank(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def retrieve_for_mode(
    query: str,
    k: int,
    mode: str,
    candidate_k: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    enable_lightweight_rerank: bool,
    rerank_bm25_weight: float,
    rerank_source_weight: float,
    rerank_keyword_weight: float,
) -> list[dict]:
    if mode == "dense":
        return retrieve_dense(query, k=k)
    if mode == "bm25":
        from bm25_retrieval import retrieve_bm25

        return retrieve_bm25(query, k=k)
    if mode == "hybrid":
        from hybrid_retrieval import retrieve_hybrid

        return retrieve_hybrid(
            query,
            k=k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            enable_lightweight_rerank=enable_lightweight_rerank,
            rerank_bm25_weight=rerank_bm25_weight,
            rerank_source_weight=rerank_source_weight,
            rerank_keyword_weight=rerank_keyword_weight,
        )
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def evaluate_example(
    example: dict,
    index: int,
    max_k: int,
    mode: str,
    candidate_k: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    enable_lightweight_rerank: bool,
    rerank_bm25_weight: float,
    rerank_source_weight: float,
    rerank_keyword_weight: float,
    chunk_lookup: dict[str, dict],
) -> dict:
    retrieved_chunks = retrieve_for_mode(
        example["question"],
        k=max_k,
        mode=mode,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        enable_lightweight_rerank=enable_lightweight_rerank,
        rerank_bm25_weight=rerank_bm25_weight,
        rerank_source_weight=rerank_source_weight,
        rerank_keyword_weight=rerank_keyword_weight,
    )
    retrieved_chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
    expected_chunk_ids = example["must_cite_chunk_ids"]
    strict_first_rank = first_relevant_rank(retrieved_chunk_ids, expected_chunk_ids)
    source_first_rank = first_relevant_rank_by_match(
        retrieved_chunk_ids,
        expected_chunk_ids,
        match_type="source",
        chunk_lookup=chunk_lookup,
    )
    neighbor_first_rank = first_relevant_rank_by_match(
        retrieved_chunk_ids,
        expected_chunk_ids,
        match_type="neighbor",
        chunk_lookup=chunk_lookup,
    )
    reciprocal_rank = 1 / strict_first_rank if strict_first_rank is not None else 0.0

    result = {
        "question_number": index,
        "question": example["question"],
        "category": example["category"],
        "expected_chunk_ids": expected_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "first_relevant_rank_strict": strict_first_rank,
        "first_relevant_rank_source": source_first_rank,
        "first_relevant_rank_neighbor": neighbor_first_rank,
        "reciprocal_rank": reciprocal_rank,
        # Backward-compatible strict fields.
        "first_relevant_rank": strict_first_rank,
        "hit": hit_at_rank(strict_first_rank, 5),
    }

    for k in EVAL_K_VALUES:
        result[f"strict_hit_at_{k}"] = hit_at_rank(strict_first_rank, k)
        result[f"source_hit_at_{k}"] = hit_at_rank(source_first_rank, k)
        result[f"neighbor_hit_at_{k}"] = hit_at_rank(neighbor_first_rank, k)
        result[f"hit_at_{k}"] = result[f"strict_hit_at_{k}"]

    return result


def build_hit_curve(per_question_results: list[dict], field_prefix: str = "hit_at") -> dict:
    total = len(per_question_results)
    hit_curve: dict[str, dict] = {}
    for k in EVAL_K_VALUES:
        key = f"{field_prefix}_{k}"
        hits = sum(1 for result in per_question_results if result[key])
        misses = total - hits
        hit_curve[key] = {
            "k": k,
            "hits": hits,
            "misses": misses,
            "rate": hits / total if total else 0.0,
        }
    return hit_curve


def mean_reciprocal_rank(per_question_results: list[dict]) -> float:
    if not per_question_results:
        return 0.0
    return sum(float(result["reciprocal_rank"]) for result in per_question_results) / len(per_question_results)


def average_first_relevant_rank(per_question_results: list[dict]) -> float | None:
    ranks = [
        int(result["first_relevant_rank"])
        for result in per_question_results
        if result["first_relevant_rank"] is not None
    ]
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def summarize_by_category(per_question_results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for result in per_question_results:
        category = result["category"]
        summary.setdefault(category, []).append(result)

    category_metrics: dict[str, dict] = {}
    for category, category_results in summary.items():
        category_metrics[category] = {
            "total": len(category_results),
            "mrr": mean_reciprocal_rank(category_results),
            "average_first_relevant_rank": average_first_relevant_rank(category_results),
            "hit_curve": build_hit_curve(category_results),
            "strict_hit_curve": build_hit_curve(category_results, "strict_hit_at"),
            "source_hit_curve": build_hit_curve(category_results, "source_hit_at"),
            "neighbor_hit_curve": build_hit_curve(category_results, "neighbor_hit_at"),
        }

    return category_metrics


def evaluate(
    examples: list[dict],
    max_k: int,
    gold_path: Path,
    mode: str,
    candidate_k: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    enable_lightweight_rerank: bool,
    rerank_bm25_weight: float,
    rerank_source_weight: float,
    rerank_keyword_weight: float,
) -> dict:
    if max_k <= 0:
        raise ValueError("max_k must be greater than 0.")
    if max_k < max(EVAL_K_VALUES):
        raise ValueError(f"max_k must be at least {max(EVAL_K_VALUES)} to compute the required metrics.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")
    if dense_weight < 0:
        raise ValueError("dense_weight must be non-negative.")
    if bm25_weight < 0:
        raise ValueError("bm25_weight must be non-negative.")

    indexed_chunks = load_json(DEFAULT_INDEXED_CHUNKS_PATH)
    indexed_chunk_lookup = build_chunk_lookup(indexed_chunks)

    per_question_results = [
        evaluate_example(
            example,
            index=index,
            max_k=max_k,
            mode=mode,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            enable_lightweight_rerank=enable_lightweight_rerank,
            rerank_bm25_weight=rerank_bm25_weight,
            rerank_source_weight=rerank_source_weight,
            rerank_keyword_weight=rerank_keyword_weight,
            chunk_lookup=indexed_chunk_lookup,
        )
        for index, example in enumerate(examples, start=1)
    ]
    total_questions = len(per_question_results)
    hit_curve = build_hit_curve(per_question_results)
    strict_hit_curve = build_hit_curve(per_question_results, "strict_hit_at")
    source_hit_curve = build_hit_curve(per_question_results, "source_hit_at")
    neighbor_hit_curve = build_hit_curve(per_question_results, "neighbor_hit_at")
    hit_at_5 = hit_curve["hit_at_5"]["rate"]
    hits_at_5 = hit_curve["hit_at_5"]["hits"]
    misses_at_5 = hit_curve["hit_at_5"]["misses"]

    return {
        "run_config": {
            "gold_path": str(gold_path),
            "max_k": max_k,
            "evaluated_k_values": list(EVAL_K_VALUES),
            "retrieval_mode": mode,
            "candidate_k": candidate_k if mode == "hybrid" else None,
            "rrf_k": rrf_k if mode == "hybrid" else None,
            "dense_weight": dense_weight if mode == "hybrid" else None,
            "bm25_weight": bm25_weight if mode == "hybrid" else None,
            "enable_lightweight_rerank": enable_lightweight_rerank if mode == "hybrid" else None,
            "rerank_bm25_weight": rerank_bm25_weight if mode == "hybrid" else None,
            "rerank_source_weight": rerank_source_weight if mode == "hybrid" else None,
            "rerank_keyword_weight": rerank_keyword_weight if mode == "hybrid" else None,
            "indexed_chunks_path": str(DEFAULT_INDEXED_CHUNKS_PATH),
        },
        "summary_metrics": {
            "total_questions": total_questions,
            "mrr": mean_reciprocal_rank(per_question_results),
            "average_first_relevant_rank": average_first_relevant_rank(per_question_results),
        },
        "hit_curve": hit_curve,
        "strict_hit_curve": strict_hit_curve,
        "source_hit_curve": source_hit_curve,
        "neighbor_hit_curve": neighbor_hit_curve,
        "by_category": summarize_by_category(per_question_results),
        "per_question_results": per_question_results,
        # Backward-compatible fields for older local inspection scripts.
        "total_questions": total_questions,
        "hits": hits_at_5,
        "misses": misses_at_5,
        "hit_at_k": hit_at_5,
        "k": 5,
    }


def print_question_result(result: dict) -> None:
    status = "HIT" if result["hit_at_5"] else "MISS"
    print(f"Question {result['question_number']}")
    print(f"Category: {result['category']}")
    print(f"Result@5: {status}")
    print(f"First relevant rank strict: {result['first_relevant_rank_strict']}")
    print(f"First relevant rank source: {result['first_relevant_rank_source']}")
    print(f"First relevant rank neighbor: {result['first_relevant_rank_neighbor']}")
    print(f"Reciprocal rank: {result['reciprocal_rank']:.4f}")
    print(f"Expected chunk IDs: {result['expected_chunk_ids']}")
    print(f"Retrieved chunk IDs: {result['retrieved_chunk_ids']}")
    print()


def print_summary(results: dict) -> None:
    summary_metrics = results["summary_metrics"]
    print("Retrieval evaluation summary")
    print("============================")
    print(f"Retrieval mode: {results['run_config']['retrieval_mode']}")
    if results["run_config"]["retrieval_mode"] == "hybrid":
        print(f"Candidate K: {results['run_config']['candidate_k']}")
        print(f"RRF K: {results['run_config']['rrf_k']}")
        print(f"Dense weight: {results['run_config']['dense_weight']}")
        print(f"BM25 weight: {results['run_config']['bm25_weight']}")
        print(f"Lightweight rerank enabled: {results['run_config']['enable_lightweight_rerank']}")
        if results["run_config"]["enable_lightweight_rerank"]:
            print(f"Rerank BM25 weight: {results['run_config']['rerank_bm25_weight']}")
            print(f"Rerank source weight: {results['run_config']['rerank_source_weight']}")
            print(f"Rerank keyword weight: {results['run_config']['rerank_keyword_weight']}")
    print(f"Total questions: {summary_metrics['total_questions']}")
    print(f"MRR: {summary_metrics['mrr']:.4f}")
    average_rank = summary_metrics["average_first_relevant_rank"]
    average_rank_text = f"{average_rank:.2f}" if average_rank is not None else "None"
    print(f"Average rank of first relevant chunk: {average_rank_text}")
    print()

    print("Strict hit curve")
    print("================")
    for key, metrics in results["hit_curve"].items():
        print(
            f"Hit@{metrics['k']}: hits={metrics['hits']}, misses={metrics['misses']}, "
            f"rate={metrics['rate']:.4f} ({metrics['rate'] * 100:.2f}%)"
        )
    print()

    print("Source hit curve")
    print("================")
    for metrics in results["source_hit_curve"].values():
        print(
            f"Source Hit@{metrics['k']}: hits={metrics['hits']}, misses={metrics['misses']}, "
            f"rate={metrics['rate']:.4f} ({metrics['rate'] * 100:.2f}%)"
        )
    print()

    print("Neighbor hit curve")
    print("==================")
    for metrics in results["neighbor_hit_curve"].values():
        print(
            f"Neighbor Hit@{metrics['k']}: hits={metrics['hits']}, misses={metrics['misses']}, "
            f"rate={metrics['rate']:.4f} ({metrics['rate'] * 100:.2f}%)"
        )
    print()

    print("Results by category")
    print("===================")
    for category, category_results in results["by_category"].items():
        hit_5 = category_results["hit_curve"]["hit_at_5"]
        hit_10 = category_results["hit_curve"]["hit_at_10"]
        print(
            f"{category}: total={category_results['total']}, "
            f"Hit@5={hit_5['rate']:.4f} ({hit_5['rate'] * 100:.2f}%), "
            f"Hit@10={hit_10['rate']:.4f} ({hit_10['rate'] * 100:.2f}%), "
            f"MRR={category_results['mrr']:.4f}"
        )
    print()

    print("Per-question results")
    print("====================")
    for result in results["per_question_results"]:
        print_question_result(result)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)


def misses_at_k(results: dict, k: int) -> list[dict]:
    key = f"hit_at_{k}"
    return [
        result
        for result in results["per_question_results"]
        if not result[key]
    ]


def save_misses(results: dict, path: Path, k: int = 5) -> None:
    misses = misses_at_k(results, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(misses, output_file, ensure_ascii=False, indent=2)


def save_all_miss_files(results: dict) -> dict[int, Path]:
    miss_paths = {
        5: DEFAULT_MISSES_AT_5_PATH,
        10: DEFAULT_MISSES_AT_10_PATH,
        20: DEFAULT_MISSES_AT_20_PATH,
    }
    for k, path in miss_paths.items():
        save_misses(results, path, k=k)
    return miss_paths


def main() -> None:
    configure_stdout()
    args = parse_args()
    gold_path = args.gold_path.resolve()
    max_k = max(args.max_k, args.k or 0, max(EVAL_K_VALUES))
    candidate_k = max(args.candidate_k, max_k)
    examples = load_jsonl(gold_path)
    results = evaluate(
        examples,
        max_k=max_k,
        gold_path=gold_path,
        mode=args.mode,
        candidate_k=candidate_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        enable_lightweight_rerank=args.enable_lightweight_rerank,
        rerank_bm25_weight=args.rerank_bm25_weight,
        rerank_source_weight=args.rerank_source_weight,
        rerank_keyword_weight=args.rerank_keyword_weight,
    )
    print_summary(results)
    save_results(results, DEFAULT_RESULTS_PATH)
    miss_paths = save_all_miss_files(results)
    print(f"Detailed results saved to: {DEFAULT_RESULTS_PATH}")
    for k, path in miss_paths.items():
        print(f"Misses@{k} saved to: {path}")


if __name__ == "__main__":
    main()
