from __future__ import annotations

import re

from bm25_retrieval import retrieve_bm25
from retrieval import retrieve as retrieve_dense


TOKEN_PATTERN = re.compile(r"[\w\u0590-\u05ff]+", flags=re.UNICODE)
DEFAULT_RERANK_BM25_WEIGHT = 0.30
DEFAULT_RERANK_SOURCE_WEIGHT = 0.20
DEFAULT_RERANK_KEYWORD_WEIGHT = 0.10


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(str(text))]


def question_bigrams(question: str) -> set[str]:
    tokens = tokenize(question)
    return {
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    }


def source_text(result: dict) -> str:
    metadata = result.get("metadata", {})
    return " ".join(
        part
        for part in (
            str(metadata.get("source", "")).strip(),
            str(metadata.get("relative_path", "")).strip(),
            str(metadata.get("category", "")).strip(),
        )
        if part
    )


def overlap_count(first: str, second: str) -> int:
    return len(set(tokenize(first)) & set(tokenize(second)))


def phrase_overlap_count(question: str, text: str) -> int:
    text_casefold = str(text).casefold()
    return sum(1 for phrase in question_bigrams(question) if phrase and phrase in text_casefold)


def apply_lightweight_rerank(
    query: str,
    fused_results: list[dict],
    bm25_results: list[dict],
    rerank_bm25_weight: float = DEFAULT_RERANK_BM25_WEIGHT,
    rerank_source_weight: float = DEFAULT_RERANK_SOURCE_WEIGHT,
    rerank_keyword_weight: float = DEFAULT_RERANK_KEYWORD_WEIGHT,
) -> list[dict]:
    bm25_rank_lookup = {
        result["chunk_id"]: rank
        for rank, result in enumerate(bm25_results, start=1)
    }

    reranked_results: list[dict] = []
    for result in fused_results:
        chunk_text = str(result.get("text", ""))
        source_overlap = overlap_count(query, source_text(result))
        keyword_overlap = overlap_count(query, chunk_text)
        phrase_overlap = phrase_overlap_count(query, chunk_text)
        bm25_rank = bm25_rank_lookup.get(result["chunk_id"])

        bm25_bonus = 0.0
        if bm25_rank is not None:
            bm25_bonus = rerank_bm25_weight * (1.0 / (5 + bm25_rank))

        source_bonus = rerank_source_weight * (min(source_overlap, 4) / 20.0)
        keyword_bonus = rerank_keyword_weight * (min(keyword_overlap + phrase_overlap, 6) / 30.0)
        total_bonus = bm25_bonus + source_bonus + keyword_bonus

        reranked_result = dict(result)
        reranked_result["base_score"] = float(result.get("score", 0.0))
        reranked_result["rerank_bonus"] = float(total_bonus)
        reranked_result["score"] = float(reranked_result["base_score"] + total_bonus)
        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("base_score", 0.0)),
        ),
        reverse=True,
    )
    return reranked_results


def reciprocal_rank_fusion(
    ranked_results: list[list[dict]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")
    if weights is None:
        weights = [1.0] * len(ranked_results)
    if len(weights) != len(ranked_results):
        raise ValueError("weights length must match ranked_results length.")

    fused_scores: dict[str, float] = {}
    result_lookup: dict[str, dict] = {}

    for results, source_weight in zip(ranked_results, weights):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                float(source_weight) * (1.0 / (rrf_k + rank))
            )
            result_lookup.setdefault(chunk_id, result)

    fused_results: list[dict] = []
    for chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        result = dict(result_lookup[chunk_id])
        result["score"] = float(score)
        fused_results.append(result)
    return fused_results


def retrieve_hybrid(
    query: str,
    k: int = 5,
    candidate_k: int = 50,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    enable_lightweight_rerank: bool = False,
    rerank_bm25_weight: float = DEFAULT_RERANK_BM25_WEIGHT,
    rerank_source_weight: float = DEFAULT_RERANK_SOURCE_WEIGHT,
    rerank_keyword_weight: float = DEFAULT_RERANK_KEYWORD_WEIGHT,
) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")

    retrieval_depth = max(k, candidate_k)
    dense_results = retrieve_dense(query, k=retrieval_depth)
    bm25_results = retrieve_bm25(query, k=retrieval_depth)
    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        rrf_k=rrf_k,
        weights=[dense_weight, bm25_weight],
    )
    if enable_lightweight_rerank:
        fused_results = apply_lightweight_rerank(
            query,
            fused_results,
            bm25_results,
            rerank_bm25_weight=rerank_bm25_weight,
            rerank_source_weight=rerank_source_weight,
            rerank_keyword_weight=rerank_keyword_weight,
        )
    return fused_results[:k]
